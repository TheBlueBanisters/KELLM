#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation.py — Evaluation entry (compatible with old scripts), supports:
- 单卡/多卡：自动检测多卡并“分片并行”评测（默认开启，可用 --auto_shard 0 关闭）
- Batch processing: when --batch_size >1, enable same-card batch inference to improve throughput
- Evaluation: after writing predictions to jsonl, delegate utils.stats for metric calculation and printing
"""

# =========================
# ===== 可调开关集中区 ====
# =========================
USE_FAST_TOKENIZER_DEFAULT: bool = False  # False is more stable (avoid protobuf/pb2 compatibility issues)
BATCH_SIZE_DEFAULT: int = 1               # Default single-item inference
SHOW_TQDM_DEFAULT: bool = True            # Show progress bar
TEMPLATE_DEFAULT: str = "alpaca"          # Template name
DTYPE_DEFAULT: str = "fp16"               # fp16 / bf16 / fp32 / auto
DEVICE_DEFAULT: int | None = None         # None=device_map="auto"；或显式 0/1/...
AUTO_SHARD_DEFAULT: bool = True           # ★ Default auto multi-GPU sharding
MERGE_TMP_KEEP: bool = False              # Whether to keep shard jsonl files after merging (default delete)
EVAL_FIRST_N_DEFAULT: int = 0            # ★ Use only first N items of test set during evaluation; 0=all

# ============== Basic dependencies ==============
import os, sys, json, argparse, subprocess, shutil
from datetime import datetime
from typing import List, Dict, Any, Iterable, Optional
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure utils package in project can be imported
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ============== Optional PEFT ==============
_PEFT_AVAILABLE = True
try:
    from peft import PeftModel
except Exception:
    _PEFT_AVAILABLE = False

# ============== Prompter prefers project internal ==============
class _FallbackPrompter:
    def __init__(self, template_name: str = "alpaca", verbose: bool = False):
        self.verbose = verbose
        self.template = {
            "prompt_input": (
                "Below is an instruction that describes a task.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            ),
            "response_split": "### Response:",
        }
    def _fmt(self, x: Any) -> str:
        if x is None: return ""
        if isinstance(x, str): return x
        try: return json.dumps(x, ensure_ascii=False)
        except Exception: return str(x)
    def generate_prompt(self, instruction: str, input: Any = None, label: str | None = None, addition: Any | None = None) -> str:
        if input is None or (isinstance(input, str) and not input.strip()):
            s = self.template["prompt_no_input"].format(instruction=instruction)
        else:
            s = self.template["prompt_input"].format(instruction=instruction, input=self._fmt(input))
        if addition: s = f"{s}\n{addition}"
        if label: s = f"{s}{label}"
        if self.verbose: print(s)
        return s
    def get_response(self, output: str) -> str:
        spl = self.template["response_split"]
        return output.split(spl, 1)[1].strip() if spl in output else output

try:
    from utils.prompter import Prompter as _ProjectPrompter  # type: ignore
except Exception:
    _ProjectPrompter = _FallbackPrompter  # type: ignore

# 解析与归一化（用于将预测过滤到候选集合内）
try:
    from utils.stats import canon as _canon  # type: ignore
except Exception:
    def _canon(s, strip_punct: bool = False):  # 最小兜底：与 utils.stats.canon 行为近似
        try:
            import unicodedata, re
            s = str(s)
            s = unicodedata.normalize("NFKC", s)
            s = s.strip().strip(" '\"[]()")
            s = s.lower().replace("_", " ")
            s = re.sub(r"\s+", " ", s).strip()
            return s
        except Exception:
            return str(s).strip().lower()

# ============== IO ==============
def iter_samples(path: str):
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line:
                    try: yield json.loads(line)
                    except Exception: pass
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for x in data:
                if isinstance(x, dict): yield x

def count_samples(path: str) -> int:
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f if _.strip())
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0

# ============== 输出解析为列表 ==============
def parse_relations(text: str) -> List[str]:
    if text is None: return []
    s = str(text).strip()
    # 1) JSON 列表
    if "[" in s and "]" in s:
        try:
            l, r = s.index("["), s.rindex("]")
            arr = json.loads(s[l:r+1])
            if isinstance(arr, list):
                out = [str(x).strip() for x in arr if str(x).strip()]
                if out: return out
        except Exception: pass
    # 2) 行条目
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    import re as _re
    cand=[]
    for ln in lines:
        m=_re.match(r"^(\d+[\.\)]|\-|\*|\•)\s*(.+)$", ln)
        if m: ln=m.group(2).strip()
        if ln: cand.append(ln)
    if len(cand)>=2: return cand
    # 3) 逗号/分号
    if "," in s or ";" in s:
        parts=[p.strip() for p in s.replace(";",",").split(",")]
        parts=[p for p in parts if p]
        if len(parts)>=2: return parts
    return [s] if s else []

def _filter_to_candidates(items: List[str], candidates: List[str]) -> List[str]:
    """
    Strictly filter parsed items to those within candidates (case/quote insensitive, keep original form).
    - Match using the same normalization rule as evaluation (utils.stats.canon)
    - Preserve order and de-duplicate
    """
    if not candidates:
        return items
    norm_to_orig = {}
    for c in candidates:
        cs = str(c).strip()
        if cs:
            norm_to_orig[_canon(cs, strip_punct=False)] = cs

    seen = set()
    kept: List[str] = []
    for it in items:
        key = _canon(it, strip_punct=False)
        orig = norm_to_orig.get(key)
        if orig is None:
            continue
        if orig in seen:
            continue
        kept.append(orig)
        seen.add(orig)
    return kept

# ============== 构建流水线 ==============
def _str2dtype(s: str):
    s=(s or "").lower()
    if s in {"fp16","float16","half"}: return torch.float16
    if s in {"bf16","bfloat16"}: return torch.bfloat16
    if s in {"fp32","float32"}: return torch.float32
    if s in {"auto",""}: return "auto"
    return torch.float16

def build_kellm_model(base_or_merged_path: str, adapter_path: str | None,
                     dtype: str, device: int | None, use_fast_tokenizer: bool):
    torch_dtype = _str2dtype(dtype)
    dev = f"cuda:{device}" if (device is not None and torch.cuda.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
    # 使用 trust_remote_code=True 以适配 Qwen 等带自定义分词/模型逻辑的权重
    tok = AutoTokenizer.from_pretrained(
        base_or_merged_path,
        use_fast=bool(use_fast_tokenizer),
        trust_remote_code=True,
    )
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        # 对于 Qwen 系列，常用 eos 作为 pad
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    # 同样信任本地模型自定义代码；不访问网络
    mdl = AutoModelForCausalLM.from_pretrained(
        base_or_merged_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if adapter_path:
        if not _PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed, but --adapter_path was provided")
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            raise FileNotFoundError(f"adapter_config.json not found: {adapter_path}")
        mdl = PeftModel.from_pretrained(mdl, adapter_path)
        try:
            mdl = mdl.merge_and_unload()
        except Exception:
            pass

    # 包装 KELLM（加载训练保存的 embeddings）
    from kellm import KELLMWithTokenTranslator
    emb_path = None
    num_prefix = 1  # 默认值
    kge_model_path = ""  # 默认值
    
    if adapter_path:
        # 1) 尝试加载训练保存的 embeddings
        cand = os.path.join(adapter_path, "embeddings.pth")
        if os.path.exists(cand):
            emb_path = cand
            # 2) 尝试从 embeddings 中读取 num_prefix（兼容 PyTorch 2.6 的 weights_only 变更）
            try:
                # 优先尝试安全白名单加载
                try:
                    from kellm import PretrainKGEmbedding as _PretrainKGEmbedding  # type: ignore
                    try:
                        import torch.serialization as _ts
                        _ts.add_safe_globals([_PretrainKGEmbedding])
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    emb_obj = torch.load(cand, map_location="cpu", weights_only=False)  # 可信本地文件
                except TypeError:
                    emb_obj = torch.load(cand, map_location="cpu")
                if hasattr(emb_obj, "num_prefix"):
                    num_prefix = int(getattr(emb_obj, "num_prefix", num_prefix) or num_prefix)
                    print(f"[KELLM] Read num_prefix from embeddings.pth: {num_prefix}")
            except Exception as e:
                print(f"[KELLM] Failed to read num_prefix from embeddings.pth, using default {num_prefix}: {e}")

        # 3) 选择 KGE 路径：优先使用环境变量；否则只选择真正包含 npy 的目录
        def _looks_like_kge_dir(d: str) -> bool:
            return (
                os.path.isfile(os.path.join(d, "entity_embedding.npy"))
                and os.path.isfile(os.path.join(d, "relation_embedding.npy"))
            )

        env_ent = os.environ.get("KGE_ENTITY_NPY", "").strip()
        env_rel = os.environ.get("KGE_RELATION_NPY", "").strip()
        if env_ent and env_rel and os.path.isfile(env_ent) and os.path.isfile(env_rel):
            # Leave empty to let load_pretrain_kge use env variables branch
            kge_model_path = ""
            print("[KELLM] Using KGE embedding files specified by environment variables")
        else:
            adapter_parent = os.path.dirname(adapter_path)
            candidates_base = [
                adapter_parent,
                os.path.dirname(adapter_parent),
                os.path.dirname(os.path.dirname(adapter_parent)),
                os.path.dirname(os.path.dirname(os.path.dirname(adapter_parent))),
            ]
            dir_names = ["CoDEx-S", "data", os.path.basename(adapter_path).split("_")[0]]
            found = None
            for base in candidates_base:
                for name in dir_names:
                    if not name:
                        continue
                    potential = os.path.abspath(os.path.join(base, name))
                    if os.path.isdir(potential) and _looks_like_kge_dir(potential):
                        found = potential
                        break
                if found:
                    break
            if found:
                kge_model_path = found
                print(f"[KELLM] Inferred kge_model path: {kge_model_path}")
    
    kellm = KELLMWithTokenTranslator(model=mdl, num_prefix=num_prefix, pretrain_emb_path=emb_path, kge_model=kge_model_path)
    # Enable decode KV cache to speed up autoregressive generation
    try:
        kellm.llama_model.config.use_cache = True
    except Exception:
        pass
    kellm.eval()
    kellm.to(dev)
    return kellm, tok, dev, torch_dtype

# ============== 多卡分片辅助 ==============
def _visible_gpu_ids() -> List[str]:
    """Get available GPU IDs from CUDA_VISIBLE_DEVICES or torch.cuda.device_count."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES","")
    if env:
        return [x for x in env.split(",") if x.strip()!=""]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []

def _spawn_subprocesses(args_ns, gpu_ids: List[str], merged_pred: str) -> None:
    """Parent process: launch a subprocess per GPU, passing shard params and output path."""
    py = sys.executable
    procs=[]
    shard_n = len(gpu_ids)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.dirname(merged_pred) or "."
    os.makedirs(out_dir, exist_ok=True)

    for idx, gid in enumerate(gpu_ids):
        shard_pred = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(merged_pred))[0]}_shard{idx}of{shard_n}.jsonl")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gid  # Each child sees exactly one GPU
        # Build command (reuse parent args)
        cmd = [
            py, __file__,
            "--data_path", args_ns.data_path,
            "--model_path", args_ns.model_path,
            "--adapter_path", args_ns.adapter_path or "",
            "--template", args_ns.template,
            "--max_new_tokens", str(args_ns.max_new_tokens),
            "--temperature", str(args_ns.temperature),
            "--top_p", str(args_ns.top_p),
            "--dtype", args_ns.dtype,
            "--use_fast_tokenizer", str(int(args_ns.use_fast_tokenizer)),
            "--batch_size", str(args_ns.batch_size),
            "--show_tqdm", "1",              # Show progress bar in each shard
            "--eval_first_n", str(args_ns.eval_first_n),  # Use only first N items (0=all)
            "--save_predictions", shard_pred,
            "--auto_shard", "0",             # Child processes should not shard again
            "--shard_index", str(idx),
            "--shard_count", str(shard_n),
            "--no_stats"                     # 子进程不算指标
        ]
        print(f"[AUTO] Launch shard process: GPU {gid} <- shard {idx}/{shard_n}")
        procs.append(subprocess.Popen(cmd, env=env))
    # Wait for all children
    code=0
    for p in procs:
        ret=p.wait()
        code = code or ret
    if code != 0:
        raise RuntimeError(f"A subprocess returned non-zero exit code: {code}")

    # Merge jsonl shards
    with open(merged_pred, "w", encoding="utf-8") as fout:
        for idx in range(shard_n):
            shard_pred = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(merged_pred))[0]}_shard{idx}of{shard_n}.jsonl")
            with open(shard_pred, "r", encoding="utf-8") as fin:
                shutil.copyfileobj(fin, fout)
            if not MERGE_TMP_KEEP:
                try: os.remove(shard_pred)
                except Exception: pass
    print(f"[AUTO] Merged {shard_n} shards -> {merged_pred}")

# ============== 主流程 ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--adapter_path", type=str, default="")
    ap.add_argument("--template", type=str, default=TEMPLATE_DEFAULT)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--dtype", type=str, default=DTYPE_DEFAULT, choices=["fp16","bf16","fp32","auto"])
    ap.add_argument("--device", type=int, default=DEVICE_DEFAULT)

    # Throughput and stability
    ap.add_argument("--use_fast_tokenizer", type=int, default=int(USE_FAST_TOKENIZER_DEFAULT))
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--show_tqdm", type=int, default=int(SHOW_TQDM_DEFAULT))
    # Use only first N test items (0=all). Works for single-card and sharded runs:
    # - Single-card: simply truncate to N
    # - Sharded: use global index mod shard_count to ensure overall coverage of first N
    ap.add_argument("--eval_first_n", type=int, default=EVAL_FIRST_N_DEFAULT,
                    help="Evaluate only the first N items; 0 uses entire test set")

    # Auto/Manual sharding for multi-GPU
    ap.add_argument("--auto_shard", type=int, default=int(AUTO_SHARD_DEFAULT), help="1=detect multi-GPU and shard automatically (default on)")
    ap.add_argument("--shard_index", type=int, default=0, help="Shard index of current process (for child processes)")
    ap.add_argument("--shard_count", type=int, default=1, help="Total number of shards (for child processes)")

    # Output predictions (auto path if not specified)
    ap.add_argument("--save_predictions", type=str, default="")
    # Evaluation controls (forwarded to utils.stats)
    ap.add_argument("--stats_snap_to_candidates", type=int, default=1)
    ap.add_argument("--stats_fuzzy", type=int, default=1)
    ap.add_argument("--stats_strip_punct", type=int, default=0)
    ap.add_argument("--stats_topn", type=int, default=20)
    ap.add_argument("--no_stats", action="store_true")
    # Native LLM input log (optional, default off to avoid huge logs)
    ap.add_argument("--log_llm_native", type=int, default=0, help="Record LLM-native input info: prompt token ids, embedding_ids, etc.")

    args = ap.parse_args()

    # ===== Parent process: auto multi-GPU sharding =====
    if args.auto_shard and args.shard_count == 1:
        gpu_ids = _visible_gpu_ids()
        if len(gpu_ids) >= 2:
            # Prepare merged output path
            if not args.save_predictions:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = os.path.join("outputs", "eval_logs", f"eval_{ts}")
                os.makedirs(run_dir, exist_ok=True)
                args.save_predictions = os.path.join(run_dir, "predictions.jsonl")
            else:
                os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
            _spawn_subprocesses(args, gpu_ids, args.save_predictions)

            # Parent only merges and runs evaluation
            if not args.no_stats:
                try:
                    from utils import stats as stats_mod
                    res = stats_mod.eval_once(
                        test_path=args.data_path,
                        pred_path=args.save_predictions,
                        snap_to_candidates=bool(args.stats_snap_to_candidates),
                        use_fuzzy=bool(args.stats_fuzzy),
                        strip_punct=bool(args.stats_strip_punct),
                        k_list=stats_mod.K_LIST,
                        topn_report=args.stats_topn,
                    )
                    print("\n================ Evaluation Summary (from utils/stats) ================")
                    stats_mod.print_summary(res)
                    # Additionally write meta.json next to predictions
                    try:
                        meta = {
                            "data_path": args.data_path,
                            "model_path": args.model_path,
                            "adapter_path": args.adapter_path,
                            "template": args.template,
                            "dtype": args.dtype,
                            "generation": {
                                "max_new_tokens": args.max_new_tokens,
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                            },
                            "batch_size": args.batch_size,
                            "shard": {
                                "auto_shard": bool(args.auto_shard),
                                "shard_index": args.shard_index,
                                "shard_count": args.shard_count,
                            },
                            "stats": res,
                        }
                        # If adapter_config.json exists, include it
                        try:
                            if args.adapter_path:
                                cfg_path = os.path.join(args.adapter_path, "adapter_config.json")
                                if os.path.isfile(cfg_path):
                                    with open(cfg_path, "r", encoding="utf-8") as f:
                                        meta["adapter_config"] = json.load(f)
                        except Exception:
                            pass
                        out_dir = os.path.dirname(args.save_predictions) or "."
                        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)
                        print(f"[INFO] Wrote meta.json -> {os.path.join(out_dir, 'meta.json')}")
                    except Exception as e:
                        print(f"[WARN] Failed to write meta.json: {e}")
                except Exception as e:
                    print(f"[WARN] Evaluation failed: {e}")
            return  # End parent process

    # ===== Child / Single-card: normal inference =====
    # Filter data by shard assignment
    def _enumerate_samples(path: str, shard_idx: int, shard_cnt: int, eval_first_n: int):
        """
        Shard and truncate by global sample index i:
        - Stop at eval_first_n when > 0
        - Keep i % shard_cnt == shard_idx, ensuring total across shards still only covers first N
        """
        for i, ex in enumerate(iter_samples(path)):
            if int(eval_first_n) > 0 and i >= int(eval_first_n):
                break
            if shard_cnt <= 1 or (i % shard_cnt) == shard_idx:
                yield ex

    # 选择 Prompter
    prompter = _ProjectPrompter(template_name=args.template, verbose=False)

    # Build KELLM model
    model, tok, device_str, torch_dtype = build_kellm_model(
        args.model_path,
        adapter_path=(args.adapter_path or None),
        dtype=args.dtype,
        device=args.device,
        use_fast_tokenizer=bool(args.use_fast_tokenizer),
    )

    # Prepare output file (create separate dir by default)
    if not args.save_predictions:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("outputs", "eval_logs", f"eval_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        args.save_predictions = os.path.join(run_dir, "predictions.jsonl")
    else:
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)

    # Counting and progress
    total_all = count_samples(args.data_path)
    # 当仅评测前 N 条时，进度上界按截断后的数量估计；分片时取上界的等分上限
    limit = int(args.eval_first_n) if int(args.eval_first_n) > 0 else total_all
    total = min(total_all, limit)
    # In child procs, progress bar upper bound is the per-shard share
    upper = total if args.shard_count==1 else (total + args.shard_count - 1)//args.shard_count
    pbar = tqdm(total=upper, desc="Running Inference", disable=not bool(args.show_tqdm))

    # Helpers
    def _as_str(x: Any) -> str:
        if isinstance(x, (str, bytes)): return x if isinstance(x, str) else x.decode("utf-8","ignore")
        try: return json.dumps(x, ensure_ascii=False)
        except Exception: return str(x)

    # Helper: build embedding_ids from sample
    def _build_ids(sample: Dict[str, Any]) -> List[int]:
        """Build ID sequence formatted as [head, tail, candidate_rel_1, ..., candidate_rel_K]"""
        seq: List[int] = []
        meta = sample.get("meta", {}) or {}
        meta_cands = meta.get("candidate_rel_ids_original", None)
        ids = sample.get("embedding_ids", None)
        
        if isinstance(ids, dict):
            h = ids.get("head_id", None)
            t = ids.get("tail_id", None)
            if h is not None: seq.append(int(h))
            if t is not None: seq.append(int(t))
            # Add all candidate relations
            if isinstance(meta_cands, list):
                seq.extend(int(x) for x in meta_cands)
            else:
                cand = ids.get("candidate_rel_ids", []) or []
                seq.extend(int(x) for x in cand)
        elif isinstance(ids, list) and len(ids) >= 3:
            # Format: [head, relation, tail]
            seq.append(int(ids[0]))  # head
            seq.append(int(ids[2]))  # tail
            
            # Add all candidate relations (from meta)
            if isinstance(meta_cands, list) and len(meta_cands) > 0:
                seq.extend(int(x) for x in meta_cands)
            else:
                # If no candidate list, at least include the correct relation
                if len(ids) >= 2 and ids[1] is not None:
                    seq.append(int(ids[1]))
        
        return seq

    # Write predictions
    written=0
    with open(args.save_predictions, "w", encoding="utf-8") as fout:
        # Generate one by one to correctly inject variable-length prefixes (avoid padding-related masking issues)
        for ex in _enumerate_samples(args.data_path, args.shard_index, args.shard_count, args.eval_first_n):
            instruction = ex.get("instruction", "")
            inp = ex.get("input", "")
            gold_list = ex.get("output", []) or []

            prefix_ids = _build_ids(ex)
            if not isinstance(prefix_ids, list):
                prefix_ids = []

            prompt = prompter.generate_prompt(instruction=instruction, input=_as_str(inp))
            enc = tok(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(model.llama_model.device)
            attn_mask = enc.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(model.llama_model.device)

            eid = torch.tensor(prefix_ids, dtype=torch.long, device=model.llama_model.device).unsqueeze(0)

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                embedding_ids=eid,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tok.eos_token_id,
            )
            out_text = tok.decode(gen_ids[0], skip_special_tokens=True)

            try: model_resp = prompter.get_response(out_text)
            except Exception: model_resp = out_text
            pred_list = parse_relations(model_resp)

            # Keep only items within candidate set, drop explanatory text/noise
            # Keep only items within candidate set, drop explanatory text/noise
            cand_list = []
            try:
                if isinstance(inp, dict):
                    raw_c = inp.get("candidates")
                    if isinstance(raw_c, list):
                        cand_list = [str(x) for x in raw_c]
            except Exception:
                cand_list = []
            if cand_list:
                pred_list = _filter_to_candidates(pred_list, cand_list)

            rec = {"instruction": instruction, "input": inp, "gold": gold_list,
                   "prediction": pred_list, "raw_output": out_text}

            if int(args.log_llm_native) == 1:
                try:
                    prompt_ids = enc["input_ids"][0].tolist()
                except Exception:
                    prompt_ids = []
                try:
                    num_prefix = int(getattr(getattr(model, "embeddings", object()), "num_prefix", 0))
                except Exception:
                    num_prefix = 0
                llm_native = {
                    "prompt_ids": prompt_ids,
                    "prompt_len": len(prompt_ids),
                    "embedding_ids": prefix_ids,
                    "num_prefix": num_prefix,
                    "prefix_token_len": (len(prefix_ids) * num_prefix) if num_prefix else 0,
                    "attention_mask_len": int(attn_mask.size(-1)) if attn_mask is not None else len(prompt_ids),
                    "eos_token_id": tok.eos_token_id,
                }
                try:
                    llm_native["dtype"] = str(torch_dtype)
                except Exception:
                    pass
                try:
                    llm_native["device"] = device_str
                except Exception:
                    pass
                rec["llm_native"] = llm_native
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            pbar.update(1)
    pbar.close()
    print(f"\n[INFO] Wrote predictions: {args.save_predictions} (records={written}; shard {args.shard_index}/{args.shard_count}; eval_first_n={int(args.eval_first_n)})")

    # No eval here for single-card/child; parent will merge and evaluate
    if not args.no_stats and args.shard_count==1:
        try:
            from utils import stats as stats_mod
            res = stats_mod.eval_once(
                test_path=args.data_path,
                pred_path=args.save_predictions,
                snap_to_candidates=bool(args.stats_snap_to_candidates),
                use_fuzzy=bool(args.stats_fuzzy),
                strip_punct=bool(args.stats_strip_punct),
                k_list=stats_mod.K_LIST,
                topn_report=args.stats_topn,
            )
            print("\n================ Evaluation Summary (from utils/stats) ================")
            stats_mod.print_summary(res)
            # Also write meta.json for single-card/child
            try:
                meta = {
                    "data_path": args.data_path,
                    "model_path": args.model_path,
                    "adapter_path": args.adapter_path,
                    "template": args.template,
                    "dtype": args.dtype,
                    "generation": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    },
                            "batch_size": args.batch_size,
                            "eval_first_n": int(args.eval_first_n),
                    "batch_size": args.batch_size,
                    "eval_first_n": int(args.eval_first_n),
                    "shard": {
                        "auto_shard": bool(args.auto_shard),
                        "shard_index": args.shard_index,
                        "shard_count": args.shard_count,
                    },
                    "stats": res,
                }
                try:
                    if args.adapter_path:
                        cfg_path = os.path.join(args.adapter_path, "adapter_config.json")
                        if os.path.isfile(cfg_path):
                            with open(cfg_path, "r", encoding="utf-8") as f:
                                meta["adapter_config"] = json.load(f)
                except Exception:
                    pass
                out_dir = os.path.dirname(args.save_predictions) or "."
                with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[INFO] Wrote meta.json -> {os.path.join(out_dir, 'meta.json')}")
            except Exception as e:
                print(f"[WARN] Failed to write meta.json: {e}")
        except Exception as e:
            print(f"[WARN] Evaluation failed: {e}")

if __name__ == "__main__":
    main()
