# -*- coding: utf-8 -*-
"""
train_kellm.py - KELLM + LoRA training script (local model + external validation set + early stopping)
 - Preserve original calling methods and functionality
 - Support A100 optimization (TF32)
 - Support BF16 control (environment variable BF16=1)
 - Optional DDP distributed training
 - Only save LoRA weights
 - New: external validation set (valid_data_path), early stopping (EarlyStopping), print specific epoch/step when early stopping occurs
"""

import os
import sys
import json
import random
from typing import List, Dict, Any, Optional

import fire
import torch
import transformers
import shutil
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # Use Auto* to be compatible with LLaMA/Qwen architectures
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback


# =========================
# A100-friendly optimization
# =========================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================
# Utility functions
# =========================
def _normalize_label(y):
    """
    Normalize output in samples to strings to avoid Prompter errors (.got list.).
    - Convert list to multi-line string (non-str use JSON)
    - Other non-str use JSON
    - None returns empty string
    """
    if y is None:
        return ""
    if isinstance(y, list):
        return "\n".join(
            s if isinstance(s, str) else json.dumps(s, ensure_ascii=False) for s in y
        )
    if not isinstance(y, str):
        return json.dumps(y, ensure_ascii=False)
    return y


def _normalize_prefix_ids(sample: Dict[str, Any], num_prefix: int) -> List[int]:
    """
    Build KoPA prefix id list (no truncation/padding here; collator aligns):
      - First two are entities: [head, tail]
      - Followed by all candidate relation IDs
    Returns: [head, tail, candidate_rel_1, candidate_rel_2, ..., candidate_rel_K]
    """
    seq: List[int] = []

    meta = sample.get("meta", {}) or {}
    meta_cands = meta.get("candidate_rel_ids_original", None)

    ids = sample.get("embedding_ids", None)
    if isinstance(ids, dict):
        h = ids.get("head_id", None)
        t = ids.get("tail_id", None)
        if h is not None:
            seq.append(int(h))
        if t is not None:
            seq.append(int(t))
        # 添加所有候选关系
        if isinstance(meta_cands, list):
            seq.extend(int(x) for x in meta_cands)
        else:
            cand = ids.get("candidate_rel_ids", []) or []
            seq.extend(int(x) for x in cand)
    elif isinstance(ids, list) and len(ids) >= 3:  # 格式：[head, relation, tail]
        # 提取 head 和 tail
        seq.append(int(ids[0]))  # head
        seq.append(int(ids[2]))  # tail
        
        # 添加所有候选关系（从 meta 中获取）
        if isinstance(meta_cands, list) and len(meta_cands) > 0:
            seq.extend(int(x) for x in meta_cands)
        else:
            # 如果没有候选关系列表，至少加入正确的关系
            if len(ids) >= 2 and ids[1] is not None:
                seq.append(int(ids[1]))

    return seq


# =========================
# Print-only EarlyStop info callback (does not change training behavior)
# =========================
class EarlyStopPrinter(TrainerCallback):
    """
    - Log best metric, step, and epoch
    - When bad_count reaches patience, print the step/epoch of early stop trigger
    Note: Actual EarlyStopping is performed by HuggingFace's EarlyStoppingCallback.
    """
    def __init__(self, metric_name: str, greater_is_better: bool, patience: int):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.patience = patience
        self.best_metric: Optional[float] = None
        self.best_step: Optional[int] = None
        self.best_epoch: Optional[float] = None
        self.bad_count: int = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.metric_name not in metrics:
            # Common path: use eval_loss
            if self.metric_name == "eval_loss" and "eval_loss" in metrics:
                current = metrics["eval_loss"]
            else:
                return
        else:
            current = metrics[self.metric_name]

        epoch_val = getattr(state, "epoch", None)  # Could be float in steps-based schedule
        step_val = state.global_step

        improved = False
        if self.best_metric is None:
            improved = True
        else:
            improved = current > self.best_metric if self.greater_is_better else current < self.best_metric

        if improved:
            self.best_metric = current
            self.best_step = step_val
            self.best_epoch = epoch_val
            self.bad_count = 0
            print(f"[EarlyStopPrinter] New best {self.metric_name}={current:.6f} "
                  f"at step={step_val}, epoch={epoch_val if epoch_val is not None else 'n/a'}")
        else:
            self.bad_count += 1
            print(f"[EarlyStopPrinter] No improvement on {self.metric_name} ({current:.6f}). "
                  f"bad_count={self.bad_count}/{self.patience} at step={step_val}, "
                  f"epoch={epoch_val if epoch_val is not None else 'n/a'}")

            if self.bad_count >= self.patience:
                # Print the moment when patience is reached; actual stop handled by HF callback
                print(f"[EarlyStopPrinter] Early stopping patience reached at step={step_val}, "
                      f"epoch={epoch_val if epoch_val is not None else 'n/a'}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.best_metric is not None:
            print(f"[EarlyStopPrinter] Best {self.metric_name}={self.best_metric:.6f} "
                  f"at step={self.best_step}, epoch={self.best_epoch if self.best_epoch is not None else 'n/a'}")
        if getattr(state, "best_model_checkpoint", None):
            print(f"[EarlyStopPrinter] Best model checkpoint: {state.best_model_checkpoint}")


# =========================
# 主训练函数
# =========================
def train(
    # ========= 必填参数 =========
    base_model: str = "",  # 本地模型目录，例如 "/path/to/llama-3-8b-instruct"
    data_path: str = "data/yago3-10-train_with_multihop.json",
    output_dir: str = "./kopa-rank-lora",

    # ========= 验证数据 =========
    valid_data_path: str = "",     # 指向你“本地划分好的验证集”文件（.json/.jsonl）。若存在则优先使用
    val_set_size: int = 0,         # 兼容保留：仅当未提供 valid_data_path 时，才用它从训练集切分

    # ========= 训练超参（集中可改动：新旧参数对照在此）=========
    # OLD defaults (仅注释保留)：
    #   batch_size=16          # 总 batch（旧值）
    #   micro_batch_size=4     # 单卡 batch（旧值）
    #   grad_accum_steps=None  # 旧行为：按 batch_size//micro_batch_size 自动计算
    # NEW defaults（更省显存，更稳）：
    batch_size: int = 128,            # NEW: 总 batch（示例：2×GPU * micro=64 * accum=1）
    micro_batch_size: int = 64,       # NEW: 单卡 batch（充分利用 48G 显存）
    grad_accum_steps: Optional[int] = 1,  # NEW: 显式梯度累积步数（旧=自动按比例）
    
    

    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1_000_000,   # 超长截断上限

    # ========= LoRA 参数 =========
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,

    # ========= KELLM 特定参数 =========
    num_prefix: int = 1,                # 每个实体/关系对应的前缀数量
    kge_model: str = "data/YAGO3-10.pth",

    # ========= 其它设置 =========
    train_on_inputs: bool = True,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "alpaca",

    # ========= 模型家族（适配 LLaMA / Qwen） =========
    model_family: str = "llama",           # llama | qwen（用于选择默认 LoRA 目标层及 pad 策略）

    # ========= 训练哪些模块 =========
    train_lora: bool = True,                 # 训练 LoRA（默认开）
    train_kellm: bool = True,                 # 训练 KELLM Token Translator（默认开）

    # ========= 早停/最佳模型（新增，可调）=========
    use_early_stopping: bool = True,           # 启用早停（仅在有验证集时生效）
    early_stopping_patience: int = 3,          # 连续多少次评估未提升则停止
    early_stopping_threshold: float = 0.0,     # EarlyStoppingCallback 的提升阈值
    load_best_model_at_end: bool = True,       # 训练结束后加载最优模型（仅在有验证集时生效）
    metric_for_best_model: str = "eval_loss",  # 最优指标（默认 eval_loss）
    greater_is_better: bool = False,           # eval_loss 越小越好

):
    assert base_model, "Please specify --base_model (本地模型目录)"

    # 精度选择（与 train.sh 的 PRECISION 对齐）：auto|bf16|fp16|fp32
    prec = str(os.environ.get("PRECISION", "auto")).lower()
    if prec == "bf16":
        use_bf16, use_fp16 = True, False
    elif prec == "fp16":
        use_bf16, use_fp16 = False, True
    elif prec == "fp32":
        use_bf16, use_fp16 = False, False
    else:  # auto
        bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        use_bf16, use_fp16 = (bf16_ok, not bf16_ok)

    # 设定随机种子（来自环境变量 SEED，默认 42）
    try:
        seed_val = int(os.environ.get("SEED", "42"))
    except Exception:
        seed_val = 42
    try:
        import numpy as _np
        _np.random.seed(seed_val)
    except Exception:
        pass
    try:
        random.seed(seed_val)
    except Exception:
        pass
    try:
        if hasattr(transformers, "set_seed"):
            transformers.set_seed(seed_val)
    except Exception:
        pass

    # 打印参数（仅主进程）
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"[train] base_model={base_model}\n"
            f"[train] data_path={data_path}\n"
            f"[train] valid_data_path={valid_data_path}\n"
            f"[train] output_dir={output_dir}\n"
            f"[train] num_prefix={num_prefix} (2 entities + K candidates)\n"
            f"[train] kge_model={kge_model}\n"
            f"[train] bf16={use_bf16}\n"
        )

    # =========================
    # DDP 处理
    # =========================
    # 依据“新参数块”优先确定有效的梯度累积步数；为空则回退为按比例计算
    gradient_accumulation_steps = (
        int(grad_accum_steps)
        if (grad_accum_steps is not None and int(grad_accum_steps) > 0)
        else max(1, batch_size // micro_batch_size)
    )
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

    # =========================
    # 模型 & tokenizer（本地文件，不访问网络）
    # =========================
    if use_bf16:
        dtype = torch.bfloat16
    elif use_fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    # 使用 trust_remote_code=True：以便加载 Qwen 等自定义建模代码（本地模型，无联网）
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # Tokenizer 同样使用 Auto*；Qwen 模型常要求 trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        # Qwen 系列通常以 eos 充当 pad；LLaMA 则使用 id=0（<pad>）
        if str(model_family).lower().startswith("qwen"):
            tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        else:
            tokenizer.pad_token_id = 0

    from utils.prompter import Prompter
    from kellm import KELLMWithTokenTranslator
    prompter = Prompter(prompt_template_name)

    # -------------------------
    # tokenize 函数
    # -------------------------
    def tokenize(prompt: str, add_eos: bool = True):
        """
        prompt -> token 序列
         - cutoff_len>0 时按 max_length 截断
         - cutoff_len<=0 时关闭截断
         - add_eos=True 且未被截断时加 eos
        """
        if cutoff_len and cutoff_len > 0:
            out = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                add_eos
                and out["input_ids"]
                and out["input_ids"][-1] != tokenizer.eos_token_id
                and len(out["input_ids"]) < cutoff_len
            ):
                out["input_ids"].append(tokenizer.eos_token_id)
                out["attention_mask"].append(1)
        else:
            out = tokenizer(prompt, truncation=False, padding=False, return_tensors=None)
            if add_eos and out["input_ids"] and out["input_ids"][-1] != tokenizer.eos_token_id:
                out["input_ids"].append(tokenizer.eos_token_id)
                out["attention_mask"].append(1)

        out["labels"] = out["input_ids"].copy()
        return out

    # -------------------------
    # 数据集映射函数
    # -------------------------
    def map_fn(dp: Dict[str, Any]):
        full_prompt = prompter.generate_prompt(
            dp.get("instruction", ""),
            dp.get("input", None),
            _normalize_label(dp.get("output", None)),
        )
        tk = tokenize(full_prompt, add_eos=add_eos_token)

        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(dp.get("instruction", ""), dp.get("input", None))
            tk_user = tokenize(user_prompt, add_eos=add_eos_token)
            user_len = len(tk_user["input_ids"])
            if add_eos_token:
                user_len -= 1
            tk["labels"] = [-100] * user_len + tk["labels"][user_len:]

        # 不再用 num_prefix 截断候选，保留全部 id；填充在 collator 内进行
        tk["embedding_ids"] = _normalize_prefix_ids(dp, 0)
        return tk

    # =========================
    # LoRA & KELLM 包装
    # =========================
    # 若未显式提供，按家族设置默认 LoRA 目标模块
    # 若用户未显式传入，按模型家族选择 LoRA 目标层集合
    if not lora_target_modules:
        fam = str(model_family).lower()
        if fam.startswith("qwen"):
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        else:
            lora_target_modules = ["q_proj", "v_proj"]

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    kellm_model = KELLMWithTokenTranslator(
        model=model,
        num_prefix=num_prefix,
        kge_model=kge_model,
        dim_llm=model.config.hidden_size,
    )
    kellm_model.train()

    # 为兼容 HuggingFace Trainer 在加载/保存时的属性访问，补齐非 PreTrainedModel 的必要属性
    # 避免在 load_best_model_at_end 阶段触发：AttributeError: '<model>' object has no attribute '_keys_to_ignore_on_save'
    if not hasattr(kellm_model, "_keys_to_ignore_on_save"):
        kellm_model._keys_to_ignore_on_save = None
    if not hasattr(kellm_model, "_keys_to_ignore_on_load_missing"):
        kellm_model._keys_to_ignore_on_load_missing = []
    if not hasattr(kellm_model, "_keys_to_ignore_on_load_unexpected"):
        kellm_model._keys_to_ignore_on_load_unexpected = []

    # =========================
    # 冻结/解冻模块
    # =========================
    if not train_lora:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = False
    if not train_kellm and hasattr(kellm_model, "embeddings") and hasattr(kellm_model.embeddings, "token_translator"):
        for p in kellm_model.embeddings.token_translator.parameters():
            p.requires_grad = False

    # 可选：打印可训练参数统计
    try:
        total, trainable = 0, 0
        for p in kellm_model.parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[params] trainable={trainable:,} / total={total:,} ({trainable/total*100:.2f}%)")
            print(f"[params] train_lora={train_lora} train_kellm={train_kellm}")
    except Exception:
        pass

    # 默认关闭梯度检查点以提升吞吐（需要更多显存）。如需开启，将 _enable_gc 改为 True。
    _enable_gc = False
    if _enable_gc:
        if hasattr(kellm_model, "llama_model") and hasattr(kellm_model.llama_model, "gradient_checkpointing_enable"):
            kellm_model.llama_model.gradient_checkpointing_enable()
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    # =========================
    # 数据集加载（优先使用本地外置验证集）
    # =========================
    if data_path.endswith((".json", ".jsonl")):
        raw = load_dataset("json", data_files=data_path)
    else:
        raw = load_dataset(data_path)

    # 优先：valid_data_path 存在 => 使用外置验证集
    has_external_val = bool(valid_data_path) and os.path.exists(valid_data_path)

    if has_external_val:
        if valid_data_path.endswith((".json", ".jsonl")):
            raw_val = load_dataset("json", data_files=valid_data_path)
        else:
            raw_val = load_dataset(valid_data_path)
        train_data = raw["train"].shuffle().map(map_fn)
        # datasets 的 "json" 加载默认把文件放到 "train" split
        val_data = raw_val["train"].shuffle().map(map_fn)
        print(f"[data] Using external validation file: {valid_data_path}")
    else:
        # 兼容原逻辑：仅当提供 val_set_size>0 时，从训练集切分
        if val_set_size > 0:
            spl = raw["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            train_data = spl["train"].shuffle().map(map_fn)
            val_data = spl["test"].shuffle().map(map_fn)
            print(f"[data] Using split validation from train, size={val_set_size}")
        else:
            train_data = raw["train"].shuffle().map(map_fn)
            val_data = None
            print("[data] No validation set provided (no external file, val_set_size=0).")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # -------------------------
    # 自定义 collator
    # -------------------------
    def data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        base = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
        # 仅保留张量相关字段给基础 collator，避免字符串字段触发 pad 错误
        kept_keys = {"input_ids", "attention_mask", "labels"}
        base_features = [{k: v for k, v in f.items() if k in kept_keys} for f in features]
        batch = base(base_features)
        # 对齐本 batch 内的 embedding_ids 长度（右侧 pad 为 -1）
        lists = [list(f.get("embedding_ids", []) or []) for f in features]
        max_len = max((len(v) for v in lists), default=0)
        mat = [v + ([-1] * (max_len - len(v))) for v in lists]
        batch["embedding_ids"] = torch.tensor(mat, dtype=torch.long)
        return batch

    # =========================
    # 断点恢复
    # =========================
    if resume_from_checkpoint:
        ckpt = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(resume_from_checkpoint, "adapter_model.bin")
        resume_from_checkpoint = False if not os.path.exists(ckpt) else resume_from_checkpoint

        if ckpt and os.path.exists(ckpt):
            print(f"[resume] from {ckpt}")
            adapters = torch.load(ckpt, map_location="cpu")
            set_peft_model_state_dict(model, adapters)

    # =========================
    # 评估/保存/早停/最佳模型 —— 仅当存在验证集时启用
    # =========================
    has_val = val_data is not None
    evaluation_strategy = "steps" if has_val else "no"  # 与 save_strategy 对齐
    save_strategy = "steps"

    # 若没有验证集，强制关闭早停 & 最佳模型加载，避免策略不匹配报错
    eff_use_early_stopping = bool(use_early_stopping and has_val)
    eff_load_best_model_at_end = bool(load_best_model_at_end and has_val)

    # 训练前设置缓存与编译
    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        if hasattr(kellm_model, "llama_model") and hasattr(kellm_model.llama_model, "config"):
            kellm_model.llama_model.config.use_cache = False
    except Exception:
        pass

    # torch.compile (可关闭)：编译 kellm_model（训练实际用的模型）
    if (
        torch.__version__ >= "2"
        and sys.platform != "win32"
        and os.environ.get("TORCHDYNAMO_DISABLE", "0") != "1"
        and not ddp  # DDP 下禁用 compile，避免 torch._dynamo/inductor 报错
    ):
        try:
            kellm_model = torch.compile(kellm_model)
        except Exception:
            pass

    # =========================
    # Trainer & TrainingArguments
    # =========================
    trainer = transformers.Trainer(
        model=kellm_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=max(1, gradient_accumulation_steps),
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=use_fp16,
            bf16=use_bf16,
            max_grad_norm=1.0,
            logging_steps=50,
            # 兼容本环境 transformers 版本：使用 eval_strategy 字段
            eval_strategy=evaluation_strategy,
            eval_steps=2000,
            save_strategy=save_strategy,
            save_steps=2000,
            save_total_limit=2,
            report_to=[],
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            output_dir=output_dir,
            dataloader_num_workers=16,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            label_names=["labels"],
            # ★ 最佳模型加载（仅在有验证集时开启，防止策略不匹配报错）
            load_best_model_at_end=eff_load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
        ),
        data_collator=data_collator,
        # ★ 回调设置：仅当有验证集且启用早停时注册
        callbacks=(
            [EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            ), EarlyStopPrinter(
                metric_name=metric_for_best_model,
                greater_is_better=greater_is_better,
                patience=early_stopping_patience,
            )] if eff_use_early_stopping else []
        ),
    )

    # HuggingFace 有时会因 use_cache=True 警告梯度检查点，这里按你原逻辑关闭（再保底一次）
    try:
        model.config.use_cache = False
    except Exception:
        pass

    # 保存 LoRA 权重（按你原逻辑）
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # torch.compile (可关闭)
    if torch.__version__ >= "2" and sys.platform != "win32" and os.environ.get("TORCHDYNAMO_DISABLE", "0") != "1":
        model = torch.compile(model)

    # =========================
    # 开始训练
    # =========================
    print(f"[config] eval={evaluation_strategy}, save={save_strategy}, "
          f"early_stop={eff_use_early_stopping}, load_best_model_at_end={eff_load_best_model_at_end}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 评估（若存在验证集）
    if has_val:
        eval_metrics = trainer.evaluate()
        print("[Eval] metrics:", eval_metrics)

    # 保存
    model.save_pretrained(output_dir)
    # 仍然额外保存 KELLM 的 embeddings（包含 token translator）
    torch.save(kellm_model.embeddings, os.path.join(output_dir, "embeddings.pth"))
    print("[OK] saved to:", output_dir)

    # 兼容评测脚本/用户期望：将适配器文件也同步到最新的 checkpoint-* 目录中
    try:
        ckpt_dirs = [
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if ckpt_dirs:
            def _ckpt_step(name: str) -> int:
                try:
                    return int(name.split("-")[-1])
                except Exception:
                    return -1
            latest = max(ckpt_dirs, key=_ckpt_step)
            dst = os.path.join(output_dir, latest)
            for fname in ("adapter_config.json", "adapter_model.bin", "embeddings.pth"):
                src = os.path.join(output_dir, fname)
                if os.path.exists(src):
                    try:
                        shutil.copy(src, os.path.join(dst, fname))
                    except Exception as _e:
                        print(f"[WARN] copy {fname} -> {dst} failed: {_e}")
            print(f"[OK] also mirrored adapter files into {dst}")
    except Exception as e:
        print(f"[WARN] failed to mirror adapter files into checkpoint dir: {e}")


# =========================
# CLI 接口
# =========================
if __name__ == "__main__":
    fire.Fire(train)
