# -*- coding: utf-8 -*-
"""TREC/BEIR-compatible IR metrics, implemented locally (pytrec_eval has no wheel on this box).

Semantics deliberately match `trec_eval` / `pytrec_eval` as used by BEIR, so numbers are
comparable to published NFCorpus results:

  nDCG@k : LINEAR gains (gain = graded relevance, NOT 2^rel - 1), discount log2(rank+1).
           IDCG@k from the ideal ordering of ALL judged docs for that query, truncated at k.
           (trec_eval's ndcg_cut_k. Unretrieved/unjudged docs contribute gain 0.)
  P@k    : |relevant in top k| / k                      (relevant := graded score > 0)
  Recall@k: |relevant in top k| / |all relevant in qrels|
  AP     : (1/R) * sum_{ranks of relevant retrieved} precision@rank, R = |all relevant|
           (trec_eval's `map`: relevant docs never retrieved contribute 0.)

A run is {qid: {doc_id: score}}; higher score = better. Ties broken deterministically by doc_id
so results are reproducible.
"""
from __future__ import annotations

import math
from collections import defaultdict


def rank_docs(scored: dict) -> list:
    """{doc_id: score} -> doc_ids sorted by score desc, ties broken by doc_id asc (deterministic)."""
    return [d for d, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))]


def dcg(gains) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked, rel, k) -> float:
    gains = [rel.get(d, 0) for d in ranked[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def precision_at_k(ranked, rel, k) -> float:
    if k == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / k


def recall_at_k(ranked, rel, k) -> float:
    n_rel = sum(1 for v in rel.values() if v > 0)
    if n_rel == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / n_rel


def average_precision(ranked, rel) -> float:
    n_rel = sum(1 for v in rel.values() if v > 0)
    if n_rel == 0:
        return 0.0
    hits = 0
    s = 0.0
    for i, d in enumerate(ranked):
        if rel.get(d, 0) > 0:
            hits += 1
            s += hits / (i + 1)
    return s / n_rel


def reciprocal_rank(ranked, rel) -> float:
    for i, d in enumerate(ranked):
        if rel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


DEFAULT_KS = (5, 10, 20, 100)


def evaluate_run(run: dict, qrels: dict, ks=DEFAULT_KS, restrict_to_qrels_queries=True):
    """Mean metrics over queries + per-query detail.

    Queries present in `qrels` but MISSING from `run` are scored as 0 (they count against the
    system) -- this is the standard, honest protocol: a system that returns nothing for a query
    does not get to skip it.
    """
    qids = sorted(qrels) if restrict_to_qrels_queries else sorted(set(qrels) | set(run))
    per_q = {}
    for qid in qids:
        rel = qrels.get(qid, {})
        if not any(v > 0 for v in rel.values()):
            continue  # no positives -> metrics undefined, excluded (report the count separately)
        ranked = rank_docs(run.get(qid, {}))
        m = {"AP": average_precision(ranked, rel), "RR": reciprocal_rank(ranked, rel),
             "n_retrieved": len(ranked)}
        for k in ks:
            m[f"nDCG@{k}"] = ndcg_at_k(ranked, rel, k)
            m[f"P@{k}"] = precision_at_k(ranked, rel, k)
            m[f"Recall@{k}"] = recall_at_k(ranked, rel, k)
        per_q[qid] = m
    if not per_q:
        return {}, {}
    keys = next(iter(per_q.values())).keys()
    mean = {k: sum(v[k] for v in per_q.values()) / len(per_q) for k in keys}
    mean["MAP"] = mean.pop("AP")
    mean["MRR"] = mean.pop("RR")
    mean["n_queries"] = len(per_q)
    return mean, per_q


def format_table(rows, ks=(10, 100), extra=("MAP", "MRR")):
    """rows: list of (name, mean_dict) -> aligned text table."""
    cols = [f"nDCG@{ks[0]}", f"P@5", f"P@{ks[0]}", f"Recall@{ks[0]}", f"Recall@{ks[-1]}"] + list(extra)
    w = max(len(n) for n, _ in rows) + 2
    out = [" " * w + "".join(f"{c:>12}" for c in cols)]
    out.append("-" * (w + 12 * len(cols)))
    for name, m in rows:
        out.append(f"{name:<{w}}" + "".join(f"{m.get(c, float('nan')):>12.4f}" for c in cols))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# self-test: hand-computed values
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # qrels: d1 rel=2, d2 rel=1, d3 rel=0 (judged non-relevant), d4 unjudged
    rel = {"d1": 2, "d2": 1, "d3": 0}
    ranked = ["d3", "d1", "d4", "d2"]  # ranks 1..4

    # DCG@4 = 0/log2(2) + 2/log2(3) + 0/log2(4) + 1/log2(5)
    exp_dcg = 0 + 2 / math.log2(3) + 0 + 1 / math.log2(5)
    # ideal ordering of judged gains [2,1,0] -> IDCG@4 = 2/log2(2) + 1/log2(3) + 0
    exp_idcg = 2 / math.log2(2) + 1 / math.log2(3)
    exp_ndcg = exp_dcg / exp_idcg
    got = ndcg_at_k(ranked, rel, 4)
    assert abs(got - exp_ndcg) < 1e-12, (got, exp_ndcg)

    # P@2 = 1/2 (d1 relevant at rank 2); Recall@2 = 1/2 (1 of 2 relevant found)
    assert abs(precision_at_k(ranked, rel, 2) - 0.5) < 1e-12
    assert abs(recall_at_k(ranked, rel, 2) - 0.5) < 1e-12
    # AP: relevant at ranks 2 and 4 -> (1/1... ) = ((1/2) + (2/4)) / 2 = 0.5
    assert abs(average_precision(ranked, rel) - 0.5) < 1e-12
    # RR: first relevant at rank 2 -> 0.5
    assert abs(reciprocal_rank(ranked, rel) - 0.5) < 1e-12

    # perfect ranking -> nDCG 1.0
    assert abs(ndcg_at_k(["d1", "d2", "d3"], rel, 10) - 1.0) < 1e-12
    # missing query counts as zero
    mean, per_q = evaluate_run({}, {"q1": rel}, ks=(10,))
    assert mean["nDCG@10"] == 0.0 and mean["n_queries"] == 1

    # queries with no positives are excluded
    mean2, _ = evaluate_run({"q1": {"d1": 1.0}}, {"q1": rel, "q2": {"dx": 0}}, ks=(10,))
    assert mean2["n_queries"] == 1

    # deterministic tie-breaking
    assert rank_docs({"b": 1.0, "a": 1.0, "c": 2.0}) == ["c", "a", "b"]

    print("metrics.py self-test PASSED (nDCG/P/Recall/AP/RR match hand-computed trec_eval values)")
