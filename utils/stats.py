# utils/stats.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中文评测组件（Raw/Aligned 双口径 + 丰富诊断）
- Raw：只做解析与文本归一化
- Aligned：先将预测序列吸附（严格等价→模糊）到候选集合，再比较
- 输出中文汇总，便于人工研读；同时提供 eval_once() 返回结构化 dict，便于程序化处理
"""

# =========================
# ===== 可调参数（集中） ===
# =========================
SNAP_TO_CANDIDATES_DEFAULT: bool = True
USE_FUZZY_DEFAULT: bool = True
STRIP_PUNCT_DEFAULT: bool = False
JACCARD_THRESH: float = 0.90
RATIO_THRESH: float = 0.92
K_LIST = [1, 3, 5, 10]
TOPN_REPORT = 20
CANDIDATE_KEYS = [
    "candidates", "relations", "relation_candidates", "candidate_relations",
    "options", "choices", "relation_options", "rel_options"
]
AUTO_TEST_PATTERNS = [
    "CoDeX-S_test_with_multihop.json",
    "*test*.json", "*test*.jsonl",
]
AUTO_PRED_PATTERNS = ["predictions_*.jsonl"]

CAND_BUCKETS = [
    (0, 0, "0（无候选）"),
    (1, 10, "1–10"),
    (11, 20, "11–20"),
    (21, 30, "21–30"),
    (31, 50, "31–50"),
    (51, 10**9, ">50"),
]

# =========================
# ===== 依赖导入 ==========
# =========================
import argparse
import difflib
import glob
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

# =========================
# ===== 文本归一化 =========
# =========================
_WS_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
LEADING_LABEL_RE = re.compile(r"^\s*(?:relation|predicate|label)\s*[:：-]\s*", re.I)

def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def canon(s: Any, strip_punct: bool = STRIP_PUNCT_DEFAULT) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = strip_accents(s)
    s = s.strip().strip(" '\"[]()")
    if s.endswith(","):
        s = s[:-1]
    s = LEADING_LABEL_RE.sub("", s)
    s = s.replace("\\\"", "\"").replace("\\'", "'")
    s = s.lower().replace("_", " ")
    if strip_punct:
        s = PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

# =========================
# ===== 读取/对齐 =========
# =========================
def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if "\n" in text and not text.startswith("["):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        if "instances" in data and isinstance(data["instances"], list):
            return data["instances"]
        return [data]
    raise RuntimeError(f"Unsupported JSON in {path}")

def auto_find_file(patterns: Sequence[str], root: str) -> str:
    for pat in patterns:
        fs = glob.glob(os.path.join(root, "**", pat), recursive=True)
        if fs:
            fs = sorted(fs, key=lambda p: os.path.getmtime(p), reverse=True)
            return fs[0]
    return ""

def extract_candidates_from_input(inp: Any) -> List[str]:
    obj: Optional[Dict[str, Any]] = None
    if isinstance(inp, dict):
        obj = inp
    elif isinstance(inp, str) and inp.strip():
        try:
            maybe = json.loads(inp)
            if isinstance(maybe, dict):
                obj = maybe
        except Exception:
            pass
    if not isinstance(obj, dict):
        return []
    for k in CANDIDATE_KEYS:
        if k in obj and isinstance(obj[k], list):
            return [canon(x) for x in obj[k] if str(x).strip()]
    return []

# =========================
# ===== 预测解析 ==========
# =========================
PARSE_LIST   = "LIST"
PARSE_JSON   = "JSON_SUBSTR"
PARSE_LINES  = "LINES"
PARSE_SPLIT  = "SPLIT"
PARSE_FALLBK = "FALLBACK"

PARSE_MODE_ZH = {
    PARSE_LIST:   "原生列表",
    PARSE_JSON:   "JSON子串",
    PARSE_LINES:  "分行条目",
    PARSE_SPLIT:  "逗号/分号分割",
    PARSE_FALLBK: "兜底（整体一项）",
}

def order_dedup(seq: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def parse_pred_field(pred_field: Any, strip_punct: bool) -> Tuple[List[str], str, int]:
    """
    返回：(规范化后的预测列表[顺序去重], 解析模式, 原始元素计数估计)
    """
    def _canon_all(xs: Sequence[Any]) -> List[str]:
        return order_dedup([canon(x, strip_punct=strip_punct) for x in xs if str(x).strip()])

    if pred_field is None:
        return [], PARSE_FALLBK, 0

    if isinstance(pred_field, list):
        xs = [x for x in pred_field if str(x).strip()]
        return _canon_all(xs), PARSE_LIST, len(xs)

    s = str(pred_field).strip()
    if not s:
        return [], PARSE_FALLBK, 0

    m = re.search(r"\[[\s\S]*?\]", s)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                xs = [x for x in arr if str(x).strip()]
                return _canon_all(xs), PARSE_JSON, len(xs)
        except Exception:
            pass

    lines = []
    for ln in s.splitlines():
        ln = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", ln)
        if ln.strip():
            lines.append(ln)
    if len(lines) >= 2:
        xs = [x for x in lines if str(x).strip()]
        return _canon_all(xs), PARSE_LINES, len(xs)

    parts = [p for p in re.split(r"[;,]", s) if p.strip()]
    if parts:
        return _canon_all(parts), PARSE_SPLIT, len(parts)

    return _canon_all([s]), PARSE_FALLBK, 1

def has_artifact_token(x: str) -> bool:
    return any(tok in x for tok in ('["', '"]', '\\"', "\\'", "],", "[", "]"))

# =========================
# ===== 吸附到候选 =========
# =========================
def snap_one(pred: str, cands: Sequence[str], use_fuzzy: bool) -> Tuple[str, bool]:
    for c in cands:
        if pred == c:
            return c, True
    if not use_fuzzy or not cands:
        return pred, False
    best = (None, -1.0)
    A = set(pred.split())
    for c in cands:
        B = set(c.split())
        j = (len(A & B) / len(A | B)) if (A or B) else 1.0
        r = difflib.SequenceMatcher(a=pred, b=c).ratio()
        sc = max(j, r)
        if sc > best[1]:
            best = (c, sc)
    if best[0] is not None and (best[1] >= JACCARD_THRESH or best[1] >= RATIO_THRESH):
        return best[0], True
    return pred, False

def snap_list(preds: Sequence[str], cands: Sequence[str], use_fuzzy: bool) -> Tuple[List[str], int, bool]:
    out = []
    snapped_cnt = 0
    snapped_any = False
    for p in preds:
        sp, ok = snap_one(p, cands, use_fuzzy)
        out.append(sp)
        if ok:
            snapped_cnt += 1
            snapped_any = True
    out = order_dedup(out)
    return out, snapped_cnt, snapped_any

# =========================
# ===== 评测与统计 =========
# =========================
def rank_of(gold: str, preds: Sequence[str]) -> Optional[int]:
    for i, p in enumerate(preds, 1):
        if p == gold:
            return i
    return None

def aggregate_metrics(ranks: List[Optional[int]], k_list: Sequence[int]) -> Dict[str, float]:
    N = len(ranks)
    if N == 0:
        base = {"acc@1": 0.0, "mrr": 0.0}
        base.update({f"hits@{k}": 0.0 for k in k_list})
        return base
    acc1 = sum(1 for r in ranks if r == 1) / N
    mrr  = sum((1.0/r) for r in ranks if r is not None) / N
    hits = {k: sum(1 for r in ranks if (r is not None and r <= k)) / N for k in k_list}
    out = {"acc@1": acc1, "mrr": mrr}
    out.update({f"hits@{k}": hits[k] for k in k_list})
    return out

def describe(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"数量": 0, "均值": 0.0, "中位数": 0.0, "标准差": 0.0,
                "最小": 0.0, "P25": 0.0, "P75": 0.0, "P90": 0.0, "P95": 0.0, "最大": 0.0}
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    def pct(p):
        k = max(0, min(n-1, int(round((p/100.0)*(n-1)))))
        return float(vals_sorted[k])
    return {
        "数量": n,
        "均值": float(mean(vals_sorted)),
        "中位数": float(median(vals_sorted)),
        "标准差": float(pstdev(vals_sorted)) if n > 1 else 0.0,
        "最小": float(vals_sorted[0]),
        "P25": pct(25),
        "P75": pct(75),
        "P90": pct(90),
        "P95": pct(95),
        "最大": float(vals_sorted[-1]),
    }

def bucket_for_size(sz: int) -> str:
    for lo, hi, name in CAND_BUCKETS:
        if lo <= sz <= hi:
            return name
    return f">{CAND_BUCKETS[-1][1]}"

def build_gold_index(test_items: List[Dict[str, Any]], strip_punct: bool) -> Dict[Tuple[str, str], Dict[str, Any]]:
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ex in test_items:
        instr = canon(ex.get("instruction", ""), strip_punct)
        inp = ex.get("input", "")
        if not isinstance(inp, str):
            try:
                inp_raw = json.dumps(inp, ensure_ascii=False, sort_keys=True)
            except Exception:
                inp_raw = str(inp)
        else:
            inp_raw = inp
        inp_norm = canon(inp_raw, strip_punct)

        out = ex.get("output", [])
        if isinstance(out, list) and out:
            gold_raw = out[0]
        elif isinstance(out, str):
            gold_raw = out
        else:
            gold_raw = ""
        gold = canon(gold_raw, strip_punct)

        cands = extract_candidates_from_input(ex.get("input", ""))
        idx[(instr, inp_norm)] = {"gold": gold, "gold_raw": gold_raw, "cands": cands}
    return idx

def eval_once(
    test_path: str,
    pred_path: str,
    snap_to_candidates: bool = SNAP_TO_CANDIDATES_DEFAULT,
    use_fuzzy: bool = USE_FUZZY_DEFAULT,
    strip_punct: bool = STRIP_PUNCT_DEFAULT,
    k_list: Sequence[int] = K_LIST,
    topn_report: int = TOPN_REPORT,
) -> Dict[str, Any]:

    tests = read_json_or_jsonl(test_path)
    preds = read_json_or_jsonl(pred_path)
    gold_idx = build_gold_index(tests, strip_punct)

    # 累积器
    n_eval = 0
    n_no_key = 0
    n_no_gold = 0
    n_has_cands = 0
    n_gold_not_in_cands = 0
    n_artifact_items = 0
    n_empty_raw = 0
    n_empty_aligned = 0
    n_oov_aligned = 0

    parse_mode_counter = Counter()
    cand_bucket_stats = defaultdict(lambda: {"N":0, "ranks_raw":[], "ranks_ali":[], "len_raw":[], "len_ali":[], "dedup_rate": []})

    # 吸附诊断
    total_snapped = 0
    items_with_snap = 0

    # 频次统计
    top1_raw_counter = Counter()
    top1_ali_counter = Counter()
    confusions_counter = Counter()  # (gold -> top1_aligned)

    lens_raw, lens_ali, dedup_rates = [], [], []
    ranks_raw, ranks_ali = [], []
    rank_hist_raw, rank_hist_ali = Counter(), Counter()

    def make_key(row: Dict[str, Any]) -> Tuple[str, str]:
        instr = canon(row.get("instruction", ""), strip_punct)
        inp = row.get("input", "")
        if not isinstance(inp, str):
            try:
                inp_raw = json.dumps(inp, ensure_ascii=False, sort_keys=True)
            except Exception:
                inp_raw = str(inp)
        else:
            inp_raw = inp
        return (instr, canon(inp_raw, strip_punct))

    for row in preds:
        key = make_key(row)
        meta = gold_idx.get(key)
        gold = None
        cands: List[str] = []

        if meta is None:
            n_no_key += 1
        else:
            gold = meta["gold"]
            cands = meta["cands"]

        if not gold:
            fb = row.get("gold", [])
            if isinstance(fb, list) and fb:
                gold = canon(fb[0], strip_punct)
            elif isinstance(fb, str) and fb.strip():
                gold = canon(fb, strip_punct)

        if not gold:
            n_no_gold += 1
            continue

        pred_parsed, parse_mode, raw_count_est = parse_pred_field(row.get("prediction"), strip_punct)
        parse_mode_counter[parse_mode] += 1

        if any(has_artifact_token(x) for x in pred_parsed):
            n_artifact_items += 1

        # Raw
        rank_r = rank_of(gold, pred_parsed)
        ranks_raw.append(rank_r)
        rank_hist_raw[rank_r if rank_r is not None else -1] += 1
        l_raw = len(pred_parsed)
        lens_raw.append(l_raw)
        if l_raw == 0:
            n_empty_raw += 1

        # Aligned
        if snap_to_candidates and cands:
            n_has_cands += 1
            cands = order_dedup(cands)
            if gold not in cands:
                n_gold_not_in_cands += 1
            snapped, snapped_cnt, snapped_any = snap_list(pred_parsed, cands, use_fuzzy)
            if snapped_any:
                items_with_snap += 1
                total_snapped += snapped_cnt
            l_ali = len(snapped)
            lens_ali.append(l_ali)
            if l_ali == 0:
                n_empty_aligned += 1
            if rank_of(gold, snapped) is None and any(x not in cands for x in snapped[:max(k_list) or 1]):
                n_oov_aligned += 1
            rank_a = rank_of(gold, snapped)
            ranks_ali.append(rank_a)
            rank_hist_ali[rank_a if rank_a is not None else -1] += 1
            if raw_count_est > 0:
                dedup_rates.append(max(0.0, (raw_count_est - l_ali) / raw_count_est))
            else:
                dedup_rates.append(0.0)
            if l_raw > 0:
                top1_raw_counter[pred_parsed[0]] += 1
            if l_ali > 0:
                top1_ali_counter[snapped[0]] += 1
                if rank_a != 1:
                    confusions_counter[(gold, snapped[0])] += 1
            bname = bucket_for_size(len(cands))
            s = cand_bucket_stats[bname]
            s["N"] += 1
            s["ranks_raw"].append(rank_r)
            s["ranks_ali"].append(rank_a)
            s["len_raw"].append(l_raw)
            s["len_ali"].append(l_ali)
            if raw_count_est > 0:
                s["dedup_rate"].append(max(0.0, (raw_count_est - l_ali) / raw_count_est))
        else:
            ranks_ali.append(rank_r)
            rank_hist_ali[rank_r if rank_r is not None else -1] += 1
            lens_ali.append(l_raw)
            if l_raw == 0:
                n_empty_aligned += 1
            if l_raw > 0:
                top1_raw_counter[pred_parsed[0]] += 1
                top1_ali_counter[pred_parsed[0]] += 1
                if rank_r != 1:
                    confusions_counter[(gold, pred_parsed[0])] += 1
            bname = bucket_for_size(len(cands) if cands else 0)
            s = cand_bucket_stats[bname]
            s["N"] += 1
            s["ranks_raw"].append(rank_r)
            s["ranks_ali"].append(rank_r)
            s["len_raw"].append(l_raw)
            s["len_ali"].append(l_raw)
            if raw_count_est > 0:
                s["dedup_rate"].append(max(0.0, (raw_count_est - l_raw) / raw_count_est))

        n_eval += 1

    metrics_raw = aggregate_metrics(ranks_raw, k_list)
    metrics_ali = aggregate_metrics(ranks_ali, k_list)

    def ratio(a, b): return (a / b) if b else 0.0
    desc_len_raw = describe([float(x) for x in lens_raw])
    desc_len_ali = describe([float(x) for x in lens_ali])
    desc_dedup   = describe([float(x) for x in dedup_rates])

    bucket_report = {}
    for name, stat in cand_bucket_stats.items():
        bucket_report[name] = {
            "样本数": stat["N"],
            "Raw指标": aggregate_metrics(stat["ranks_raw"], k_list),
            "Aligned指标": aggregate_metrics(stat["ranks_ali"], k_list),
            "长度分布（Raw）": describe([float(x) for x in stat["len_raw"]]),
            "长度分布（Aligned）": describe([float(x) for x in stat["len_ali"]]),
            "去重率分布": describe([float(x) for x in stat["dedup_rate"]]),
        }

    res = {
        "计分样本数": n_eval,
        "无法对齐样本数（键不匹配）": n_no_key,
        "缺少gold样本数": n_no_gold,
        "含候选集合样本占比": ratio(
            sum(1 for name, s in cand_bucket_stats.items() if name != "0（无候选）" and s["N"]>0), n_eval),
        "gold不在候选集合占比（仅含候选样本）": ratio(
            n_gold_not_in_cands,
            max(1, sum(1 for name, s in cand_bucket_stats.items() if name != "0（无候选）" and s["N"]>0))
        ),
        "预测含脏符号样本占比": ratio(n_artifact_items, n_eval),
        "空预测样本占比（Raw）": ratio(n_empty_raw, n_eval),
        "空预测样本占比（Aligned）": ratio(n_empty_aligned, n_eval),
        "对齐后Top-K仍含非候选项占比": ratio(n_oov_aligned, n_eval),

        "准确率指标（Raw）": metrics_raw,
        "准确率指标（Aligned）": metrics_ali,
        "排名直方图（Raw，-1=未命中）": dict(sorted(Counter({**Counter(ranks_raw)}).items(),
                                    key=lambda x: (x[0] if x[0] is not None else 10**9))),
        "排名直方图（Aligned，-1=未命中）": dict(sorted(Counter({**Counter(ranks_ali)}).items(),
                                    key=lambda x: (x[0] if x[0] is not None else 10**9))),

        "序列长度分布（Raw）": desc_len_raw,
        "序列长度分布（Aligned）": desc_len_ali,
        "去重率分布": desc_dedup,

        "解析模式计数": {},  # 下方填充为中文键名
        "平均每条吸附数量（仅含候选样本）": ratio(
            total_snapped,
            max(1, sum(stat["N"] for name, stat in cand_bucket_stats.items() if name != "0（无候选）"))
        ),
        "发生过吸附的样本占比": ratio(items_with_snap, n_eval),

        "Top-1预测频次（Raw）": Counter(top1_raw_counter).most_common(topn_report),
        "Top-1预测频次（Aligned）": Counter(top1_ali_counter).most_common(topn_report),
        "常见混淆对（gold→对齐后Top-1）": [((g, p), c) for (g, p), c in Counter(confusions_counter).most_common(topn_report)],

        "按候选规模分桶报告": bucket_report,

        "运行参数": {
            "snap_to_candidates": snap_to_candidates,
            "use_fuzzy": use_fuzzy,
            "strip_punct": strip_punct,
            "JACCARD_THRESH": JACCARD_THRESH,
            "RATIO_THRESH": RATIO_THRESH,
            "K_LIST": list(k_list),
            "TOPN_REPORT": topn_report,
        },
    }
    # 解析模式计数（中文化）
    res["解析模式计数"] = {
        PARSE_MODE_ZH.get(k, k): v for k, v in dict(parse_mode_counter).items()
    }
    return res

# =========================
# ===== 打印中文汇总 =======
# =========================
def _print_desc_block(title: str, d: Dict[str, float]):
    print(f"\n===== {title} =====")
    order = ["数量","均值","中位数","标准差","最小","P25","P75","P90","P95","最大"]
    print("  " + "， ".join([f"{k}={d[k]:.3f}" if isinstance(d.get(k, 0), (int, float)) else f"{k}={d.get(k)}"
                          for k in order]))

def _print_metrics_block(title: str, metrics: Dict[str, float], hist: Dict[Any, int], k_list: Sequence[int]):
    print(f"\n===== 准确率指标（{title}） =====")
    print(f"Acc@1={metrics['acc@1']:.4f}  MRR={metrics['mrr']:.4f}")
    for k in k_list:
        print(f"Hits@{k:<2}={metrics[f'hits@{k}']:.4f}", end="  ")
    print("\n排名直方图（-1=未命中）：", hist)

def print_summary(res: Dict[str, Any], k_list: Sequence[int] = K_LIST):
    print("\n===== 概览（诊断） =====")
    for key in [
        "计分样本数",
        "无法对齐样本数（键不匹配）",
        "缺少gold样本数",
        "含候选集合样本占比",
        "gold不在候选集合占比（仅含候选样本）",
        "预测含脏符号样本占比",
        "空预测样本占比（Raw）",
        "空预测样本占比（Aligned）",
        "对齐后Top-K仍含非候选项占比",
        "平均每条吸附数量（仅含候选样本）",
        "发生过吸附的样本占比",
    ]:
        v = res.get(key, None)
        if v is None:
            continue
        if isinstance(v, float):
            print(f"{key:28s}: {v:.4f}")
        else:
            print(f"{key:28s}: {v}")

    _print_desc_block("序列长度统计（Raw）", res["序列长度分布（Raw）"])
    _print_desc_block("序列长度统计（Aligned）", res["序列长度分布（Aligned）"])
    _print_desc_block("去重率统计", res["去重率分布"])

    print("\n===== 解析模式统计 =====")
    pm = res["解析模式计数"]
    total_pm = sum(pm.values()) or 1
    for m, c in pm.items():
        pct = c / total_pm * 100.0
        print(f"  {m:14s}: {c:6d}  ({pct:5.1f}%)")

    _print_metrics_block("Raw", res["准确率指标（Raw）"], res["排名直方图（Raw，-1=未命中）"], k_list)
    _print_metrics_block("Aligned", res["准确率指标（Aligned）"], res["排名直方图（Aligned，-1=未命中）"], k_list)

    print("\n===== Top-1 预测频次（Raw） =====")
    for s, c in res["Top-1预测频次（Raw）"]:
        print(f"  {s} : {c}")
    print("\n===== Top-1 预测频次（Aligned） =====")
    for s, c in res["Top-1预测频次（Aligned）"]:
        print(f"  {s} : {c}")

    print("\n===== 常见混淆对（gold → 对齐后Top-1） =====")
    for (g, p), c in res["常见混淆对（gold→对齐后Top-1）"]:
        print(f"  {g} → {p} : {c}")

    print("\n===== 按候选规模分桶报告 =====")
    for name, br in res["按候选规模分桶报告"].items():
        print(f"[候选规模：{name}]  样本数={br['样本数']}")
        mr = br["Raw指标"]; ma = br["Aligned指标"]
        print(f"  Raw    : Acc@1={mr['acc@1']:.4f}, MRR={mr['mrr']:.4f}, " +
              "， ".join([f"Hit@{k}={mr[f'hits@{k}']:.4f}" for k in k_list]))
        print(f"  Aligned: Acc@1={ma['acc@1']:.4f}, MRR={ma['mrr']:.4f}, " +
              "， ".join([f"Hit@{k}={ma[f'hits@{k}']:.4f}" for k in k_list]))
        def fmt_desc(d):
            order = ["数量","均值","中位数","标准差","最小","P25","P75","P90","P95","最大"]
            return "， ".join([f"{k}={d[k]:.3f}" if isinstance(d.get(k,0),(int,float)) else f"{k}={d.get(k)}"
                               for k in order])
        print("  长度（Raw）    :", fmt_desc(br["长度分布（Raw）"]))
        print("  长度（Aligned）:", fmt_desc(br["长度分布（Aligned）"]))
        print("  去重率         :", fmt_desc(br["去重率分布"]))

# =========================
# ===== CLI 入口 ==========
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, default="", help="测试集 JSON/JSONL 路径")
    ap.add_argument("--pred", type=str, default="", help="预测 JSONL 路径（predictions_*.jsonl）")
    ap.add_argument("--snap_to_candidates", type=int, default=1 if SNAP_TO_CANDIDATES_DEFAULT else 0,
                    help="是否吸附到候选集合（1/0）")
    ap.add_argument("--fuzzy", type=int, default=1 if USE_FUZZY_DEFAULT else 0, help="是否启用模糊吸附（1/0）")
    ap.add_argument("--strip_punct", type=int, default=1 if STRIP_PUNCT_DEFAULT else 0, help="是否去标点（1/0）")
    ap.add_argument("--topn", type=int, default=TOPN_REPORT, help="TopN 列表展示数量")
    args = ap.parse_args()

    root = os.getcwd()
    test_path = args.test or auto_find_file(AUTO_TEST_PATTERNS, root)
    pred_path = args.pred or auto_find_file(AUTO_PRED_PATTERNS, root)

    if not test_path or not os.path.exists(test_path):
        print(f"[错误] 未找到测试集文件：--test <path> ；搜索根目录={root}")
        return
    if not pred_path or not os.path.exists(pred_path):
        print(f"[错误] 未找到预测文件：--pred <path> ；搜索根目录={root}")
        return

    res = eval_once(
        test_path=test_path,
        pred_path=pred_path,
        snap_to_candidates=bool(args.snap_to_candidates),
        use_fuzzy=bool(args.fuzzy),
        strip_punct=bool(args.strip_punct),
        k_list=K_LIST,
        topn_report=args.topn,
    )
    print_summary(res, k_list=K_LIST)

if __name__ == "__main__":
    main()
