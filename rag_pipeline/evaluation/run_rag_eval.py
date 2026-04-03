"""
批量评测 RAG：读 JSONL 金标集，对每条跑检索+生成，输出聚合指标与明细 JSON。

用法（在仓库根目录）:
  python -m rag_pipeline.evaluation.run_rag_eval \\
    --benchmark rag_pipeline/evaluation/sample_benchmark.jsonl \\
    --output rag_pipeline/evaluation/last_report.json \\
    --load_in_4bit

金标 JSONL 每行字段（除 question 外均可选）:
  - question: 必填
  - id, tags: 可选，便于筛选与报表（评测脚本会原样保留在 items 中）
  - gold_answer: 标准答案（用于 token F1 / 规范化精确匹配）；可省略表示只评检索
  - gold_chunk_id: 与 ece269_meta.json 中某条 chunk_id 一致（检索命中）
  - gold_context_substring: 应出现在任一检索片段 text 中（大小写不敏感）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_pipeline.chat_rag import (
    build_rag_args,
    generate_answer,
    load_generator,
    load_retriever,
    resolve_index_meta,
    search,
    build_messages,
)
from rag_pipeline.evaluation.metrics import (
    aggregate_retrieval_metrics,
    answer_vs_gold_metrics,
    retrieval_hit_at_k,
    retrieval_score_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG benchmark evaluation")
    p.add_argument("--benchmark", type=str, required=True, help="JSONL 评测集路径")
    p.add_argument("--output", type=str, default="", help="写出完整报告的 JSON 路径")
    p.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--adapter_path", type=str, default=None)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--rag_root", type=str, default=None, help="默认与 chat_rag 相同")
    p.add_argument("--rag_variant", type=str, default="ece269")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.3)
    return p.parse_args()


def main() -> None:
    args_ns = parse_args()
    bench_path = Path(args_ns.benchmark)
    if not bench_path.is_file():
        raise FileNotFoundError(bench_path)

    from rag_pipeline.chat_rag import DEFAULT_RAG_DATA_ROOT

    rag_root = args_ns.rag_root or DEFAULT_RAG_DATA_ROOT
    rag_args = build_rag_args(
        base_model_name=args_ns.base_model_name,
        adapter_path=args_ns.adapter_path,
        load_in_4bit=args_ns.load_in_4bit,
        rag_root=rag_root,
        rag_variant=args_ns.rag_variant,
        top_k=args_ns.top_k,
        max_new_tokens=args_ns.max_new_tokens,
        temperature=args_ns.temperature,
        top_p=args_ns.top_p,
    )

    if rag_args.adapter_path is None:
        cand = Path(__file__).resolve().parent.parent.parent / "Qlora-result" / "checkpoint-2400"
        if cand.is_dir() and (cand / "adapter_config.json").is_file():
            rag_args.adapter_path = str(cand)

    index_file, meta_file = resolve_index_meta(rag_args)
    print("Loading retriever...")
    embed_model, index, metadata = load_retriever(rag_args, index_file, meta_file)
    print("Loading generator...")
    model, tokenizer = load_generator(rag_args)

    items: list[dict] = []
    with open(bench_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row.get("question", "").strip()
            if not q:
                continue
            retrieved = search(
                query=q,
                args=rag_args,
                embed_model=embed_model,
                index=index,
                metadata=metadata,
            )
            messages = build_messages(rag_args, q, retrieved)
            answer = generate_answer(args=rag_args, model=model, tokenizer=tokenizer, messages=messages)

            rec_m = retrieval_score_metrics(retrieved)
            ans_m = answer_vs_gold_metrics(answer, row.get("gold_answer"))
            hit_m = retrieval_hit_at_k(
                retrieved,
                gold_chunk_id=row.get("gold_chunk_id"),
                gold_context_substring=row.get("gold_context_substring"),
            )

            record = {
                "id": row.get("id", line_no),
                "question": q,
                "prediction": answer,
                "retrieved": retrieved,
                "retrieval_scores": rec_m,
                **ans_m,
                **hit_m,
            }
            if row.get("tags") is not None:
                record["tags"] = row["tags"]
            items.append(record)
            print(f"[{record['id']}] top1={rec_m['top1_score']:.4f} token_f1={ans_m.get('token_f1', 'n/a')} hit={hit_m.get('retrieval_hit', 'n/a')}")

    # 聚合
    avg_ret = aggregate_retrieval_metrics(items)
    with_gold_ans = [x for x in items if x.get("has_gold")]
    mean_f1 = (
        sum(x["token_f1"] for x in with_gold_ans) / len(with_gold_ans) if with_gold_ans else None
    )
    mean_exact = (
        sum(x["exact_norm_match"] for x in with_gold_ans) / len(with_gold_ans) if with_gold_ans else None
    )
    with_gold_ret = [x for x in items if x.get("has_gold_retrieval")]
    mean_hit = (
        sum(x["retrieval_hit"] for x in with_gold_ret) / len(with_gold_ret) if with_gold_ret else None
    )

    summary = {
        "n_items": len(items),
        "avg_retrieval": avg_ret,
        "answer_gold_count": len(with_gold_ans),
        "mean_token_f1_on_gold": mean_f1,
        "mean_exact_norm_match_on_gold": mean_exact,
        "retrieval_gold_count": len(with_gold_ret),
        "mean_retrieval_hit_on_gold": mean_hit,
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args_ns.output:

        def _truncate_retrieved(rows: list[dict], max_chars: int = 800) -> list[dict]:
            slim = []
            for c in rows:
                d = dict(c)
                t = d.get("text") or ""
                if len(t) > max_chars:
                    d["text"] = t[:max_chars] + "…"
                slim.append(d)
            return slim

        slim_items = []
        for it in items:
            d = dict(it)
            d["retrieved"] = _truncate_retrieved(d.get("retrieved") or [])
            slim_items.append(d)
        report = {"summary": summary, "items": slim_items}
        outp = Path(args_ns.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
