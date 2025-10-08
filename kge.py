# kge.py
import os
from typing import Tuple, Optional

import numpy as np
import torch


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"Embedding must be 2-D, got shape={tuple(x.shape)}")
    return x


def _ensure_2d_and_match(
    ent: torch.Tensor, rel: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    ent = _ensure_2d(ent).contiguous()
    rel = _ensure_2d(rel).contiguous()

    de, dr = ent.shape[1], rel.shape[1]
    if de == dr:
        return ent.contiguous(), rel.contiguous()

    # RotatE common: entity = 2 × relation -> duplicate and concatenate relation
    if de == 2 * dr:
        rel = torch.cat([rel, rel], dim=-1)
        return ent.contiguous(), rel.contiguous()

    # 反向极少见：关系 = 2 × 实体 -> 复制拼接实体
    if dr == 2 * de:
        ent = torch.cat([ent, ent], dim=-1)
        return ent.contiguous(), rel.contiguous()

    raise ValueError(
        f"Dim mismatch between entity ({de}) and relation ({dr}). "
        f"Supported patterns: equal or 2× (RotatE-like)."
    )


def _load_npy(path: str) -> torch.Tensor:
    arr = np.load(path, allow_pickle=False)
    return torch.from_numpy(arr).float().contiguous()


def _load_npy_pair(ent_path: str, rel_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not os.path.isfile(ent_path):
        raise FileNotFoundError(f"Entity embedding file not found: {ent_path}")
    if not os.path.isfile(rel_path):
        raise FileNotFoundError(f"Relation embedding file not found: {rel_path}")
    ent = _load_npy(ent_path)
    rel = _load_npy(rel_path)
    ent, rel = _ensure_2d_and_match(ent, rel)
    return ent, rel


def _guess_npy_in_dir(kge_dir: str) -> Tuple[str, str]:
    ent_candidates = ["entity_embedding.npy", "entities.npy", "ent_embedding.npy"]
    rel_candidates = ["relation_embedding.npy", "relations.npy", "rel_embedding.npy"]

    def _pick(base_dir: str, names: list) -> Optional[str]:
        for n in names:
            p = os.path.join(base_dir, n)
            if os.path.isfile(p):
                return p
        return None

    ent_path = _pick(kge_dir, ent_candidates)
    rel_path = _pick(kge_dir, rel_candidates)
    if not ent_path or not rel_path:
        raise FileNotFoundError(
            f"Could not find entity/relation npy in dir: {kge_dir}. "
            f"Tried {ent_candidates} and {rel_candidates}"
        )
    return ent_path, rel_path


def load_pretrain_kge(kge_model: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回：ent_embs, rel_embs 两个 2D float32 张量，最终列数相同。
      1) 若 kge_model 是目录 -> 在目录里自动找 *.npy；
      2) 否则从 env 变量 KGE_ENTITY_NPY / KGE_RELATION_NPY 读取。
    """
    if isinstance(kge_model, str) and kge_model and os.path.isdir(kge_model):
        ent_path, rel_path = _guess_npy_in_dir(kge_model)
        ent, rel = _load_npy_pair(ent_path, rel_path)
        print(f"[KGE] Loaded from dir={kge_model}: ent={tuple(ent.shape)}, rel={tuple(rel.shape)}")
        return ent, rel

    ent_from_env = os.environ.get("KGE_ENTITY_NPY", "").strip()
    rel_from_env = os.environ.get("KGE_RELATION_NPY", "").strip()
    if ent_from_env and rel_from_env:
        ent, rel = _load_npy_pair(ent_from_env, rel_from_env)
        print(f"[KGE] Loaded from env: ent={tuple(ent.shape)}, rel={tuple(rel.shape)}")
        return ent, rel

    raise ValueError(
        "Could not locate KGE embeddings. "
        "Either pass a directory path via --kge_model that contains *.npy, "
        "or export KGE_ENTITY_NPY and KGE_RELATION_NPY."
    )
