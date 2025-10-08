# kellm.py
from typing import Optional, List, Tuple, Any

import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel  # Use generic base class for compatibility with LLaMA/Qwen architectures

from kge import load_pretrain_kge


# ---------- Utility Functions ----------
def _get_llm_dim(model: PreTrainedModel) -> int:
    try:
        return model.get_input_embeddings().embedding_dim
    except Exception:
        # Try more generic fallbacks (different architectures may have different embedding names)
        try:
            return model.model.embed_tokens.embedding_dim  # Common structure: model.embed_tokens
        except Exception:
            try:
                return model.model.model.embed_tokens.embedding_dim  # LLaMA style
            except Exception as _e:
                raise AttributeError(f"Cannot infer embedding dim from model: {_e}")

def _get_llm_device_dtype(model: PreTrainedModel) -> Tuple[torch.device, torch.dtype]:
    for p in model.parameters():
        return p.device, p.dtype
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dev, (torch.float16 if dev.type == "cuda" else torch.float32)

def _shape_str(x: torch.Tensor, name: str) -> str:
    return f"{name}: shape={tuple(x.shape)}, device={x.device}, dtype={x.dtype}"

def _range_check_1d(name: str, idx: torch.Tensor, upper: int):
    if idx.dtype != torch.long:
        raise TypeError(f"{name} must be torch.long, got {idx.dtype} with shape={tuple(idx.shape)}")
    if idx.numel() == 0:
        return
    mn = int(idx.min().item())
    mx = int(idx.max().item())
    if mn < 0 or mx >= upper:
        # Extract first few out-of-bounds points for debugging
        bad = (idx < 0) | (idx >= upper)
        bad_pos = bad.nonzero(as_tuple=False)
        sample = bad_pos[:10].view(-1).tolist()
        raise RuntimeError(
            f"[IndexRangeError] {name} out of range [0, {upper-1}]: min={mn}, max={mx}. "
            f"first_bad_linear_positions={sample}  (total_bad={bad_pos.size(0)})  "
            f"(upper={upper})"
        )

def _range_check_2d(name: str, idx: torch.Tensor, upper: int):
    if idx.dtype != torch.long:
        raise TypeError(f"{name} must be torch.long, got {idx.dtype} with shape={tuple(idx.shape)}")
    if idx.numel() == 0:
        return
    mn = int(idx.min().item())
    mx = int(idx.max().item())
    if mn < 0 or mx >= upper:
        bad = (idx < 0) | (idx >= upper)
        bad_pos = bad.nonzero(as_tuple=False)  # [N, 2] -> (batch_i, col_j)
        sample = bad_pos[:10].tolist()
        raise RuntimeError(
            f"[IndexRangeError] {name} out of range [0, {upper-1}]: min={mn}, max={mx}. "
            f"first_bad_positions(B,Col)={sample}  (total_bad={bad_pos.size(0)})  "
            f"(upper={upper}, shape={tuple(idx.shape)})"
        )


# ================================ Main Module ================================
class KELLMWithTokenTranslator(nn.Module):
    """
    KELLM: Load pre-trained KGE (entity/relation) and map to LLM prefix tokens via Token Translator.
    - Dimension mismatches (e.g., RotatE's 2x) are aligned in kge.py during loading;
    - Always uses the same token translator layer here.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        num_prefix: int,
        kge_model: str = "data",
        pretrain_emb_path: Optional[str] = None,
        dim_llm: Optional[int] = None,   # Backward compatibility; auto-detect if not provided
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        # For backward compatibility: keep llama_model naming while providing generic alias llm_model.
        # All external access to underlying LLM remains available (generate/get_input_embeddings, etc.).
        self.llama_model = model
        self.llm_model = model

        # 1) KGE: loader ensures ent/rel have consistent column counts
        ent_embs, rel_embs = load_pretrain_kge(kge_model)

        # 2) LLM dim/device/dtype
        llm_dim = int(dim_llm) if dim_llm is not None else _get_llm_dim(self.llama_model)
        llm_device, llm_dtype = _get_llm_device_dtype(self.llama_model)

        # 3) Build KGE Token Translator module
        if pretrain_emb_path is None:
            print("Token Translator Trained From Scratch")
            self.embeddings = PretrainKGEmbedding(
                pretrain_ent_embs=ent_embs,
                pretrain_rel_embs=rel_embs,
                dim_llm=llm_dim,
                num_prefix=num_prefix,
                padding_idx=padding_idx,
                dtype=llm_dtype,   # Initial dtype alignment
            )
        else:
            print(f"Token Translator Load From {pretrain_emb_path}")
            # Compatible with PyTorch 2.6+ safe deserialization:
            # - embeddings.pth saves module objects (PretrainKGEmbedding), need to allow corresponding global types
            try:
                import torch.serialization as _ts
                # Allow nn.Embedding and our project's PretrainKGEmbedding
                try:
                    from torch.nn.modules.sparse import Embedding as _TorchEmbedding  # type: ignore
                    _ts.add_safe_globals([_TorchEmbedding])
                except Exception:
                    pass
                try:
                    # When this file is imported as module, class name is kellm.PretrainKGEmbedding
                    _ts.add_safe_globals([PretrainKGEmbedding])
                except Exception:
                    pass
            except Exception:
                pass

            # Local trusted file, allow weights_only=False
            try:
                self.embeddings = torch.load(pretrain_emb_path, map_location="cpu", weights_only=False)
            except TypeError:
                # Compatible with older torch.load without weights_only parameter
                self.embeddings = torch.load(pretrain_emb_path, map_location="cpu")

        # Align device/dtype with LLM (initialization phase)
        self.embeddings.to(device=llm_device, dtype=self.embeddings.target_dtype)

        # Warm up cuBLAS (small GEMM to avoid initialization on first step)
        if llm_device.type == "cuda":
            try:
                a = torch.randn(2, 2, device=llm_device, dtype=self.embeddings.target_dtype)
                b = torch.randn(2, 2, device=llm_device, dtype=self.embeddings.target_dtype)
                (a @ b).sum().item()
                torch.cuda.synchronize()
            except Exception as _e:
                print(f"[KELLM Warmup] cuBLAS warmup skipped: {_e}")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        embedding_ids: torch.LongTensor = None,  # [B,3] or [B,L] or [B]
        **kwargs,
    ):
        assert embedding_ids is not None, "embedding_ids is required for KELLMWithTokenTranslator."

        # A. Dynamically sync to current LLM device/dtype (HF training may cause cast)
        base_device, base_dtype = _get_llm_device_dtype(self.llama_model)

        # Input transfer #
        if embedding_ids.device != base_device:
            embedding_ids = embedding_ids.to(base_device, non_blocking=True)
        if input_ids.device != base_device:
            input_ids = input_ids.to(base_device, non_blocking=True)
        if attention_mask is not None and attention_mask.device != base_device:
            attention_mask = attention_mask.to(base_device, non_blocking=True)
        if labels is not None and labels.device != base_device:
            labels = labels.to(base_device, non_blocking=True)

        # Submodule sync (if HF does amp/bf16 cast externally, follow here) #
        self.embeddings.maybe_sync(device=base_device, dtype=base_dtype)

        # B. Compute KG prefix
        kg_prefix = self.embeddings(embedding_ids)  # [B, P, H] (here P is the length after "each id one prefix")
        if kg_prefix.dtype != base_dtype:
            kg_prefix = kg_prefix.to(base_dtype)
        kg_prefix = kg_prefix.contiguous()
        bsz, p_len, h_dim = kg_prefix.shape

        # B2. Compute valid prefix mask based on original embedding_ids, mask invalid/padding columns
        # Unify ids to [B,L], construct upper bounds for each column (first two=entity upper bound, rest=relation upper bound)
        ids = embedding_ids
        if ids.dim() == 1:
            ids = ids.unsqueeze(1)
        if ids.dim() == 2:
            B, L = ids.shape
            ids_long = ids.long()
            ent_upper = int(getattr(self.embeddings, "num_ent", 0))
            rel_upper = int(getattr(self.embeddings, "num_rel", 0))
            if L > 0:
                uppers = torch.full((L,), rel_upper, dtype=ids_long.dtype, device=ids_long.device)
                uppers[0] = ent_upper
                if L >= 2:
                    uppers[1] = ent_upper
                valid_cols = (ids_long >= 0) & (ids_long < uppers.unsqueeze(0))  # [B,L]
                # Expand to prefix dimension (each id corresponds to num_prefix tokens) -> [B, L*num_prefix]
                prefix_valid = valid_cols.repeat_interleave(int(self.embeddings.num_prefix), dim=1)
                # Align to kg_prefix shape, mask invalid prefix vectors
                mask_embed = prefix_valid.unsqueeze(-1).to(dtype=kg_prefix.dtype)
                if mask_embed.shape[:2] == kg_prefix.shape[:2]:
                    kg_prefix = kg_prefix * mask_embed

        # C. Text token embeddings
        tok_embed = self.llama_model.get_input_embeddings()(input_ids)  # [B, T, H]
        if tok_embed.dtype != base_dtype:
            tok_embed = tok_embed.to(base_dtype)
        tok_embed = tok_embed.contiguous()
        b2, t_len, h2 = tok_embed.shape

        # D. Strict shape check before concatenation
        if bsz != b2 or h_dim != h2:
            raise RuntimeError(
                "Prefix/Text embedding shape mismatch before cat:\n"
                f"  {_shape_str(kg_prefix, 'kg_prefix')}\n"
                f"  {_shape_str(tok_embed, 'tok_embed')}\n"
                "=> Expect same batch (dim0) and hidden (dim2)."
            )

        inputs_embeds = torch.cat([kg_prefix, tok_embed], dim=1)  # [B, P+T, H]

        # E. Concatenate prefix into attention mask (invalid prefix positions set to 0)
        if attention_mask is not None:
            prefix_mask = torch.ones((bsz, p_len), dtype=attention_mask.dtype, device=base_device)
        else:
            prefix_mask = torch.ones((bsz, p_len), dtype=torch.long, device=base_device)
        # If prefix_valid is already computed, overlay as prefix mask
        if ids.dim() == 2:
            B, L = ids.shape
            ent_upper = int(getattr(self.embeddings, "num_ent", 0))
            rel_upper = int(getattr(self.embeddings, "num_rel", 0))
            if L > 0:
                ids_long = ids.long()
                uppers = torch.full((L,), rel_upper, dtype=ids_long.dtype, device=ids_long.device)
                uppers[0] = ent_upper
                if L >= 2:
                    uppers[1] = ent_upper
                valid_cols = (ids_long >= 0) & (ids_long < uppers.unsqueeze(0))  # [B,L]
                prefix_valid = valid_cols.repeat_interleave(int(self.embeddings.num_prefix), dim=1)  # [B,P]
                prefix_mask = prefix_mask * prefix_valid.to(dtype=prefix_mask.dtype)
        if attention_mask is not None:
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
        else:
            # When not provided, construct based on prefix and text total length, text part is 1
            text_mask = torch.ones((bsz, t_len), dtype=prefix_mask.dtype, device=base_device)
            attention_mask = torch.cat([prefix_mask, text_mask], dim=-1)

        # F. Prefix positions do not participate in loss
        if labels is not None:
            prefix_labels = torch.full((bsz, p_len), fill_value=-100, dtype=torch.long, device=base_device)
            labels = torch.cat([prefix_labels, labels], dim=-1)

        # Optional: DEBUG prints (enable with KELLM_DEBUG=1)
        if os.environ.get("KELLM_DEBUG", "0") == "1":
            print("[DEBUG] " + _shape_str(kg_prefix, "kg_prefix"))
            print("[DEBUG] " + _shape_str(tok_embed, "tok_embed"))
            print("[DEBUG] " + _shape_str(inputs_embeds, "inputs_embeds"))

        return self.llama_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


    # Compatible with transformers' generate API
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """
        Forward to underlying LLM while ensuring embedding_ids take effect.
        Supports: generate(input_ids=..., attention_mask=..., embedding_ids=..., **gen_kwargs)
        """
        # 从 kwargs 中取出 embedding_ids，先走一次前向来拼接前缀到 inputs_embeds
        embedding_ids = kwargs.pop("embedding_ids", None)
        if embedding_ids is None:
            # If embedding_ids is not provided, pass through to base model
            return self.llama_model.generate(*args, **kwargs)

        # 组装必要输入，复用 forward 的逻辑以构造 inputs_embeds 与 attention_mask
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if input_ids is None:
            raise ValueError("generate requires input_ids in order to concatenate KELLM prefix.")

        # Use forward-built inputs_embeds/attention_mask, then call underlying generate
        out = self.forward(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=None,
                           embedding_ids=embedding_ids,
                           use_cache=kwargs.get("use_cache", True))
        # The above forward has already concatenated prefix into attention_mask and returned logits/loss etc.;
        # To avoid duplicate computation, manually construct inputs_embeds here and call underlying generate:
        base_device, base_dtype = _get_llm_device_dtype(self.llama_model)
        kg_prefix = self.embeddings(embedding_ids.to(base_device))
        if kg_prefix.dtype != base_dtype:
            kg_prefix = kg_prefix.to(base_dtype)
        tok_embed = self.llama_model.get_input_embeddings()(input_ids.to(base_device))
        if tok_embed.dtype != base_dtype:
            tok_embed = tok_embed.to(base_dtype)
        inputs_embeds = torch.cat([kg_prefix.contiguous(), tok_embed.contiguous()], dim=1)

        if attention_mask is not None:
            prefix_mask = torch.ones((inputs_embeds.size(0), kg_prefix.size(1)), dtype=attention_mask.dtype, device=base_device)
            attention_mask = torch.cat([prefix_mask, attention_mask.to(base_device)], dim=-1)
        else:
            attention_mask = torch.ones((inputs_embeds.size(0), inputs_embeds.size(1)), dtype=torch.long, device=base_device)

        return self.llama_model.generate(inputs_embeds=inputs_embeds,
                                         attention_mask=attention_mask,
                                         **kwargs)

# ================================ KGE -> 前缀 ================================
class PretrainKGEmbedding(nn.Module):
    """
     Aligned KGE (entity/relation have same column count), share one Token Translator:
      token_translator: Linear(pretrain_dim -> emb_dim), where emb_dim = num_prefix * dim_llm.
     In this implementation, "each id corresponds to num_prefix prefix tokens":
       - Reshape token_translator output to [*, num_prefix, dim_llm], keep all num_prefix slices.
    Supported inputs:
      - [B,3]: (h, r, t) three segments (entity/relation/entity lookup separately);
      - [B,L]: first two positions as entity, rest as relation;
      - [B]: single entity.
    """
    def __init__(
        self,
        pretrain_ent_embs: torch.FloatTensor,
        pretrain_rel_embs: torch.FloatTensor,
        dim_llm: int,
        num_prefix: int,
        padding_idx: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_prefix = num_prefix
        self.llm_dim = dim_llm
        self.emb_dim = num_prefix * dim_llm
        self.target_dtype = dtype

        # 1) 统一 dtype
        pretrain_ent_embs = pretrain_ent_embs.to(dtype=self.target_dtype).contiguous()
        pretrain_rel_embs = pretrain_rel_embs.to(dtype=self.target_dtype).contiguous()

        # 2) 冻结 embedding
        self.ent_embeddings = nn.Embedding.from_pretrained(
            pretrain_ent_embs, freeze=True, padding_idx=padding_idx
        )
        self.rel_embeddings = nn.Embedding.from_pretrained(
            pretrain_rel_embs, freeze=True, padding_idx=padding_idx
        )

        # 置零 padding 行
        with torch.no_grad():
            if padding_idx is not None and 0 <= padding_idx < self.ent_embeddings.weight.shape[0]:
                self.ent_embeddings.weight[padding_idx].zero_()
            if padding_idx is not None and 0 <= padding_idx < self.rel_embeddings.weight.shape[0]:
                self.rel_embeddings.weight[padding_idx].zero_()

        self.num_ent = self.ent_embeddings.num_embeddings
        self.num_rel = self.rel_embeddings.num_embeddings

        self.pretrain_dim = self.ent_embeddings.weight.shape[1]
        assert self.pretrain_dim == self.rel_embeddings.weight.shape[1], \
            f"Expect matched dims after loading, got ent={self.pretrain_dim} vs rel={self.rel_embeddings.weight.shape[1]}"

        # 3) 共享 Token Translator
        self.token_translator = nn.Linear(self.pretrain_dim, self.emb_dim, bias=True)
        nn.init.xavier_uniform_(self.token_translator.weight)
        if self.token_translator.bias is not None:
            nn.init.zeros_(self.token_translator.bias)
        self.token_translator.to(dtype=self.target_dtype)

        try:
            print(f"[PretrainKGEmbedding] pretrain_dim={self.pretrain_dim} -> emb_dim={self.emb_dim} "
                  f"(num_prefix={self.num_prefix}, llm={self.llm_dim}, dtype={self.target_dtype})")
        except Exception:
            pass

    # —— 训练时若外部做了 amp/bf16 cast，这里动态跟随 —— #
    def maybe_sync(self, device: torch.device, dtype: torch.dtype):
        need_move = (self.target_dtype != dtype) or any(p.device != device for p in self.parameters())
        if need_move:
            self.target_dtype = dtype
            self.to(device=device, dtype=dtype)

    def forward(self, ids: torch.LongTensor):
        """
        ids:
          - [B,3]  -> (h,r,t)  => [B, 3*num_prefix, llm_dim]
           - [B,L]  -> L entity segments   => [B, L*num_prefix, llm_dim]
           - [B]    -> 1 entity segment   => [B,   num_prefix, llm_dim]
        """
        if ids.dim() == 2:
            B, L = ids.shape

            if L == 3:
                 # Strict range check (keep error behavior for early detection of config issues)
                _range_check_1d("head_ids.flatten()", ids[:, 0].reshape(-1).cpu(), self.num_ent)
                _range_check_1d("rel_ids.flatten()", ids[:, 1].reshape(-1).cpu(), self.num_rel)
                _range_check_1d("tail_ids.flatten()", ids[:, 2].reshape(-1).cpu(), self.num_ent)

                head, relation, tail = ids[:, 0], ids[:, 1], ids[:, 2]
                h = self.ent_embeddings(head)     # [B, D]
                r = self.rel_embeddings(relation) # [B, D]
                t = self.ent_embeddings(tail)     # [B, D]

                h = self.token_translator(h)  # [B, emb_dim]
                r = self.token_translator(r)
                t = self.token_translator(t)

                 # Each id produces num_prefix prefix tokens -> [B, num_prefix, H]
                h = h.view(B, self.num_prefix, self.llm_dim)
                r = r.view(B, self.num_prefix, self.llm_dim)
                t = t.view(B, self.num_prefix, self.llm_dim)

                embs = torch.stack((h, r, t), dim=1)  # [B, 3, num_prefix, H]
                return embs.view(B, 3 * self.num_prefix, self.llm_dim)

             # ---- [B,L] general branch: treat L as L "entity/segments" ----
            # 1) Ensure long dtype
            if ids.dtype != torch.long:
                ids = ids.long()

             # 2) Column processing: first two as entity, rest as relation; lookup by column and clean out-of-bounds
            e_list = []  # One [B, D] per column
            for j in range(L):
                col = ids[:, j]
                if col.dtype != torch.long:
                    col = col.long()
                if j < 2:
                     # Entity column
                    col_work = col.clone()
                    invalid = (col_work < 0) | (col_work >= self.num_ent)
                    if invalid.any():
                        if os.environ.get("KELLM_DEBUG", "0") == "1":
                            bad_pos = invalid.nonzero(as_tuple=False).view(-1).tolist()
                            print(f"[KGE sanitize-ENT col={j}] mask {len(bad_pos)}/{col_work.numel()} invalid -> 0; first={bad_pos[:10]}")
                        col_work[invalid] = 0
                    e_list.append(self.ent_embeddings(col_work))
                else:
                     # Relation column
                    col_work = col.clone()
                    invalid = (col_work < 0) | (col_work >= self.num_rel)
                    if invalid.any():
                        if os.environ.get("KELLM_DEBUG", "0") == "1":
                            bad_pos = invalid.nonzero(as_tuple=False).view(-1).tolist()
                            print(f"[KGE sanitize-REL col={j}] mask {len(bad_pos)}/{col_work.numel()} invalid -> 0; first={bad_pos[:10]}")
                        col_work[invalid] = 0
                    e_list.append(self.rel_embeddings(col_work))

             # 3) Linear mapping -> keep all num_prefix slices -> flatten to [B, L*num_prefix, H]
            e = torch.stack(e_list, dim=1)        # [B, L, D]
            e = self.token_translator(e)                   # [B, L, emb_dim]
            e = e.view(B, L * self.num_prefix, self.llm_dim)  # [B, L*num_prefix, H]
            return e

        elif ids.dim() == 1:
            B = ids.size(0)
            if ids.dtype != torch.long:
                ids = ids.long()

            # Single entity mode also do safety cleaning (rare)
            ids_work = ids.clone()
            invalid = (ids_work < 0) | (ids_work >= self.num_ent)
            if invalid.any():
                if os.environ.get("KELLM_DEBUG", "0") == "1":
                    bad_pos = invalid.nonzero(as_tuple=False).view(-1).tolist()
                    print(f"[KGE sanitize-1D] mask {len(bad_pos)}/{ids_work.numel()} invalid ids "
                          f"to padding_idx=0; first_bad_idx={bad_pos[:10]}")
                ids_work[invalid] = 0

            e = self.ent_embeddings(ids_work)  # [B, D]
            e = self.token_translator(e)               # [B, emb_dim]
            e = e.view(B, self.num_prefix, self.llm_dim)  # [B, num_prefix, H]
            return e  # [B, num_prefix, H]

        else:
            raise ValueError(
                f"Unsupported embedding_ids shape: {tuple(ids.shape)}. "
                "Expected [B,3] (h,r,t) or [B,L] (L >= 1) or [B]."
            )

