"""RAG 评分指标（无需额外 pip 包）：检索分数、命中、答案与金标对比。"""

from __future__ import annotations

import re
import string
from typing import Any


def retrieval_score_metrics(retrieved: list[dict]) -> dict[str, float]:
    """
    仅基于检索结果自身（无需金标）：看 top-k 相似度是否集中、是否有明显断层。
    FAISS Inner Product 已归一化向量时，分数可近似理解为余弦相似度。
    """
    if not retrieved:
        return {
            "n": 0.0,
            "top1_score": 0.0,
            "mean_score": 0.0,
            "min_score": 0.0,
            "margin_top1_minus_last": 0.0,
        }
    scores = [float(r.get("score", 0.0)) for r in retrieved]
    return {
        "n": float(len(scores)),
        "top1_score": scores[0],
        "mean_score": sum(scores) / len(scores),
        "min_score": scores[-1],
        "margin_top1_minus_last": scores[0] - scores[-1],
    }


def aggregate_retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """对多条样本的 retrieval_score_metrics 做简单平均（跳过 n==0）。"""
    acc: dict[str, float] = {}
    count = 0
    for r in rows:
        m = retrieval_score_metrics(r.get("retrieved") or [])
        if m["n"] == 0:
            continue
        count += 1
        for k, v in m.items():
            acc[k] = acc.get(k, 0.0) + v
    if count == 0:
        return {k: 0.0 for k in ("n", "top1_score", "mean_score", "min_score", "margin_top1_minus_last")}
    return {k: acc[k] / count for k in acc}


def _normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    for p in string.punctuation:
        s = s.replace(p, "")
    return s


def _token_f1(pred: str, gold: str) -> float:
    pt = _normalize_answer(pred).split()
    gt = _normalize_answer(gold).split()
    if not pt or not gt:
        return 0.0
    gset = {}
    for t in gt:
        gset[t] = gset.get(t, 0) + 1
    hit = 0
    for t in pt:
        if gset.get(t, 0) > 0:
            hit += 1
            gset[t] -= 1
    precision = hit / len(pt)
    recall = hit / len(gt)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def answer_vs_gold_metrics(prediction: str, gold_answer: str | None) -> dict[str, Any]:
    """有标准答案时：规范化精确匹配、子串、token-level F1。"""
    if not gold_answer or not gold_answer.strip():
        return {"has_gold": False}
    g = gold_answer.strip()
    p = prediction.strip()
    np, ng = _normalize_answer(p), _normalize_answer(g)
    return {
        "has_gold": True,
        "exact_norm_match": int(np == ng),
        "gold_in_pred": int(ng in np) if ng else 0,
        "pred_in_gold": int(np in ng) if np else 0,
        "token_f1": round(_token_f1(p, g), 4),
    }


def retrieval_hit_at_k(
    retrieved: list[dict],
    *,
    gold_chunk_id: str | None = None,
    gold_context_substring: str | None = None,
) -> dict[str, Any]:
    """
    检索是否命中金标（任选其一）：
    - gold_chunk_id：与 chunk 元数据里的 chunk_id 一致即命中
    - gold_context_substring：任一条检索正文的子串（大小写不敏感）
    """
    if not gold_chunk_id and not (gold_context_substring and gold_context_substring.strip()):
        return {"has_gold_retrieval": False}

    hit = False
    rank: int | None = None
    needle = (gold_context_substring or "").strip().lower()

    for i, item in enumerate(retrieved):
        if gold_chunk_id and item.get("chunk_id") == gold_chunk_id:
            hit = True
            rank = i + 1
            break
        if needle:
            text = (item.get("text") or "").lower()
            if needle in text:
                hit = True
                rank = i + 1
                break

    return {
        "has_gold_retrieval": True,
        "retrieval_hit": int(hit),
        "retrieval_hit_rank": rank,
    }
