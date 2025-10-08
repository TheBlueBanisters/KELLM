KELLM: Knowledge-Enhanced LLM for Entity–Relation Prediction
============================================================

Overview
--------

KELLM augments a base Large Language Model (LLM) with structured knowledge from a pre-trained Knowledge Graph Embedding (KGE). It translates entity/relation embeddings into a sequence of learnable prefix tokens and injects them in front of the text tokens, enabling the LLM to reason over candidate relations conditioned on the head and tail entities.

This repository provides:
- A lightweight Token Translator that maps KGE vectors → LLM prefix tokens.
- Training via LoRA on top of local LLMs (e.g., Qwen/LLaMA) without internet access.
- Evaluation with multi-GPU sharding and rich metrics.
- A reference dataset layout (e.g., CoDEx-S with multi-hop evidence) and simple integration with your own data.

Key ideas
---------

- Map KGE vectors to LLM space: A single linear layer projects entity/relation embeddings to a stack of `num_prefix` hidden vectors per id, forming a learnable prefix sequence.
- Flexible prefix formats: Support `[B,3]` (head, relation, tail), `[B,L]` (first two entity columns then L relation columns), or `[B]` (single-entity prefix).
- Non-intrusive integration: The LLM remains a standard `transformers` model; inference/training is done by feeding `inputs_embeds` with the prefix concatenated.
- Data-driven candidates: At training/evaluation time, pass `[head, tail, candidate_rel_1, ..., candidate_rel_K]` so the LLM ranks from the candidate relation set.

Repository structure
--------------------

- `kellm.py`: Core module (`KELLMWithTokenTranslator`, `PretrainKGEmbedding`).
- `kge.py`: KGE loading and dimension alignment utilities (supports equal or 2× dims, e.g., RotatE).
- `train_kellm.py`: LoRA training over a local LLM with KELLM prefixes.
- `evaluation.py`: Single-/multi-GPU evaluation with shard merge and metric computation.
- `eval.sh`: Convenience launcher for evaluation with auto dataset/checkpoint resolution.
- `utils/`:
  - `prompter.py`: Template-aware prompt builder with structured-input support.
  - `stats.py`: Evaluation metrics (Raw/Aligned), diagnostics and summary printing.
- `templates/alpaca.json`: Default instruction-following template.
- `CoDEx-S/`: Example dataset and KGE embeddings (for quick start).

Environment and installation
----------------------------

Requirements
- Python 3.10+
- CUDA-capable GPU recommended

Install
```bash
python -m pip install -r requirements.txt
pip install -e .
```

Environment variables (optional)
- `PRECISION=auto|bf16|fp16|fp32` (training precision; default auto)
- `SEED=42` (global random seed)
- `TORCHDYNAMO_DISABLE=1` (disable torch.compile if needed)
- `KGE_ENTITY_NPY` and `KGE_RELATION_NPY` (absolute paths to `.npy` files if you don’t pass a KGE dir)
- `TRANSFORMERS_VERBOSITY=error`, `HF_HUB_DISABLE_PROGRESS_BARS=1` (reduce logs)

Dataset format
--------------

Each record is a dict compatible with instruction tuning. Typical keys:
- `instruction`: task instruction string
- `input`: either a raw string or a structured dict; for structured inputs we recommend:
  - `head`: head entity label
  - `tail`: tail entity label
  - `candidates`: list of candidate relation labels
  - `paths_text` or `paths[].path_text`: optional multi-hop evidence strings
- `output`: gold relation label or a list whose first element is the gold label
- `embedding_ids`: one of:
  - dict: `{head_id, tail_id, candidate_rel_ids}`
  - list `[head_id, relation_id, tail_id]` (at least the correct triple)
- `meta.candidate_rel_ids_original`: optional canonical candidate id list (preferred if present)

KGE embeddings
--------------

Provide pre-trained KGE matrices as `.npy` files:
- `entity_embedding.npy`: shape `[num_entities, pretrain_dim]`
- `relation_embedding.npy`: shape `[num_relations, pretrain_dim]`

Ways to point KELLM to your KGE:
1) Pass a directory path containing the above files via `--kge_model <DIR>`.
2) Or export absolute paths:
```bash
export KGE_ENTITY_NPY=/abs/path/to/entity_embedding.npy
export KGE_RELATION_NPY=/abs/path/to/relation_embedding.npy
```

Dimension alignment (handled automatically in `kge.py`):
- Equal dims are used as-is.
- RotatE-style `entity_dim = 2 × relation_dim` is supported (relation duplicated).
- The reverse rare case `relation_dim = 2 × entity_dim` is also supported (entity duplicated).

How prefixes are built
----------------------

`PretrainKGEmbedding` maps each id to `num_prefix` vectors of size `dim_llm` via a learned linear layer:
- Input shapes:
  - `[B,3]` → `[B, 3×num_prefix, dim_llm]` for `(head, relation, tail)`
  - `[B,L]` → `[B, L×num_prefix, dim_llm]` with first two columns treated as entity ids, the rest as relation ids
  - `[B]`   → `[B, num_prefix, dim_llm]` (single entity)
- Invalid/out-of-range ids are sanitized to padding id 0 during the general path; labels for prefix positions are masked (`-100`).

Training
--------

Minimal example (single node; local LLM path):
```bash
python train_kellm.py \
  --base_model models/Qwen2.5-3B \
  --data_path CoDEx-S/CoDEx-S_train_with_multihop.json \
  --valid_data_path CoDEx-S/CoDEx-S_valid_with_multihop.json \
  --output_dir outputs/CoDEx-S_$(date +%F_%H-%M-%S) \
  --num_prefix 1 \
  --kge_model CoDEx-S \
  --model_family qwen \
  --batch_size 128 --micro_batch_size 64 --grad_accum_steps 1 \
  --num_epochs 3 --learning_rate 3e-4 \
  --use_early_stopping True --early_stopping_patience 3
```

Notes
- LoRA target modules default by family (Qwen: attention+MLP; LLaMA: `q_proj`, `v_proj`). Override with `--lora_target_modules` if needed.
- Precision is controlled via `PRECISION` env (auto/bf16/fp16/fp32). Defaults to auto.
- Checkpoints save LoRA weights in `output_dir` and also save `embeddings.pth` for KELLM.
- For convenience, the latest `checkpoint-*` dir is mirrored with `adapter_config.json`, `adapter_model.bin`, and `embeddings.pth`.

Distributed training
- DDP is supported via standard `torchrun`/`accelerate` launchers. The script auto-adjusts gradient accumulation.
- Set `WORLD_SIZE`, `LOCAL_RANK`, etc., per your launcher.

Evaluation
----------

Quick start with the provided script:
```bash
bash eval.sh
```
This script resolves the dataset/test file, checkpoint/adapter directory, KGE paths, and launches `evaluation.py` with:
- Multi-GPU sharding when multiple GPUs are visible
- Prediction writing to `outputs/eval_logs/eval_<timestamp>/predictions.jsonl`
- Auto computation of Raw/Aligned metrics (MRR, Acc@1, Hits@K) via `utils/stats.py`

Manual invocation example:
```bash
python -W ignore evaluation.py \
  --data_path CoDEx-S/CoDEx-S_test_with_multihop.json \
  --model_path models/Qwen2.5-3B \
  --adapter_path outputs/<your_run>/checkpoint-*/ \
  --template alpaca \
  --max_new_tokens 128 --temperature 0.0 --top_p 1.0 \
  --dtype fp16 --use_fast_tokenizer 0 \
  --eval_first_n 0 \
  --save_predictions outputs/eval_logs/predictions.jsonl
```

Evaluation details
- `evaluation.py` auto-detects GPUs and spawns one subprocess per GPU when `--auto_shard 1` (default). Parent merges shards and prints metrics.
- `prediction` is parsed and snapped to the candidate set (if provided) before scoring. Diagnostics include sequence lengths, parse-mode distribution, OOV rate after snapping, and more.
- Set `--log_llm_native 1` to record prompt token ids and prefix metadata for debugging.

Inference (programmatic)
------------------------

Use the KELLM wrapper to inject prefixes at generation time:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kellm import KELLMWithTokenTranslator
import torch

tok = AutoTokenizer.from_pretrained("models/Qwen2.5-3B", trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-3B", trust_remote_code=True)

kellm = KELLMWithTokenTranslator(
    model=mdl,
    num_prefix=1,
    kge_model="CoDEx-S",  # or set KGE_* envs
    pretrain_emb_path=None,
)

prompt = "..."
enc = tok(prompt, return_tensors="pt")
eid = torch.tensor([[head_id, tail_id, *candidate_rel_ids]], dtype=torch.long)

gen = kellm.generate(
    input_ids=enc["input_ids"],
    attention_mask=enc.get("attention_mask"),
    embedding_ids=eid,
    max_new_tokens=128,
)
print(tok.decode(gen[0], skip_special_tokens=True))
```

Configuration reference
-----------------------

Important `train_kellm.py` args
- `--base_model`: local path to the LLM (e.g., `models/Qwen2.5-3B`)
- `--data_path`, `--valid_data_path`: training and validation files
- `--output_dir`: output directory
- `--num_prefix`: number of prefix tokens per id
- `--kge_model`: directory containing `entity_embedding.npy` and `relation_embedding.npy` (or leave empty and set env vars)
- `--model_family`: `llama` or `qwen` (affects defaults like pad token and LoRA targets)
- LoRA: `--lora_r`, `--lora_alpha`, `--lora_dropout`, `--lora_target_modules`
- Early stopping: `--use_early_stopping`, `--early_stopping_patience`, `--metric_for_best_model`

Important `evaluation.py` args
- `--data_path`, `--model_path`, `--adapter_path`
- `--dtype {fp16,bf16,fp32,auto}`
- `--batch_size` (same-card batching)
- `--eval_first_n` (evaluate first N items; 0 = all)
- `--auto_shard {0,1}` (multi-GPU sharding)
- `--save_predictions` (jsonl path)
- Stats controls: `--stats_snap_to_candidates`, `--stats_fuzzy`, `--stats_strip_punct`, `--stats_topn`

Reproducing paper-style experiments (CoDEx-S)
---------------------------------------------

1) Prepare KGE
```bash
# Already provided under CoDEx-S/ for quick start.
# To use your own, export absolute paths:
export KGE_ENTITY_NPY=/abs/path/to/entity_embedding.npy
export KGE_RELATION_NPY=/abs/path/to/relation_embedding.npy
```

2) Train
```bash
python train_kellm.py \
  --base_model models/Qwen2.5-3B \
  --data_path CoDEx-S/CoDEx-S_train_with_multihop.json \
  --valid_data_path CoDEx-S/CoDEx-S_valid_with_multihop.json \
  --output_dir outputs/CoDEx-S_$(date +%F_%H-%M-%S) \
  --num_prefix 1 --kge_model CoDEx-S --model_family qwen \
  --batch_size 128 --micro_batch_size 64 --grad_accum_steps 1 \
  --num_epochs 3 --learning_rate 3e-4
```

3) Evaluate
```bash
bash eval.sh
# or
python evaluation.py --data_path CoDEx-S/CoDEx-S_test_with_multihop.json \
  --model_path models/Qwen2.5-3B --adapter_path outputs/<your_run>/checkpoint-* \
  --template alpaca --max_new_tokens 128 --temperature 0.0 --top_p 1.0 --dtype fp16
```

Troubleshooting
---------------

- Dtype/device mismatch: The wrapper syncs submodules each forward, but if you see cast errors, ensure the base model and embeddings use compatible dtypes (e.g., fp16/bf16 on GPU).
- `torch.load` safety in PyTorch ≥2.6: We whitelist module types before loading `embeddings.pth`; if you still see errors, ensure you use the same project code version that saved the file.
- Prefix length vs. attention mask: We auto-concatenate a prefix mask; ensure your `embedding_ids` are shaped and padded as expected. In training, collator pads with `-1` and KELLM masks invalid columns.
- Candidates missing gold: Metrics report the rate of “gold not in candidates”; double-check data preprocessing.
- Qwen pad token: If your tokenizer lacks a pad id, we fall back to `eos_token_id` (Qwen) or `0` (LLaMA).

FAQ
---

Q: Can I use `[B,3]` ids at inference?
A: Yes. For candidate ranking, prefer `[head, tail, candidate_rel_1, ...]`. For single-triple conditioning, pass `[head, relation, tail]`.

Q: How many prefix tokens per id should I use?
A: Start with `--num_prefix 1` for efficiency. Larger values increase prefix length linearly.

Q: Do I need internet to load models?
A: No. Scripts load local models only (`local_files_only=True`).

Citation
--------

If you find this project helpful, please consider citing your paper associated with this repository. Example BibTeX (edit with your details):
```bibtex
@article{your_kellm_paper,
  title={Knowledge-Enhanced Large Language Models for Entity–Relation Prediction},
  author={Your Name and Co-authors},
  journal={...},
  year={2025}
}
```

License
-------

This project inherits licenses of the base models and datasets you use. See `LICENSE` files included with models/datasets as applicable.


