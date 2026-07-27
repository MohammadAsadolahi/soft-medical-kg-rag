# -*- coding: utf-8 -*-
"""Paired bootstrap significance testing for IR runs.

Retrieval systems are compared on the same queries, and per-query metric variance on a benchmark of a
few hundred queries is large -- large enough that differences of a couple of nDCG points are routinely
noise. Reporting two means and declaring the larger one better is the most common way to draw a wrong
conclusion from a benchmark.

The paired bootstrap tests the null hypothesis that two systems are equivalent, by resampling
*queries* (with replacement) and asking how often the observed difference in means could arise by
chance. Pairing matters: both systems are scored on the identical resample, so the query-difficulty
variance that dominates the raw spread cancels.

Reported per comparison:
  * mean of each system, and the difference
  * a 95% percentile confidence interval on the difference
  * a two-sided p-value
  * win/loss/tie counts, which show whether a mean difference comes from a broad shift or a few
    outlier queries -- two very different situations with the same mean.
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap(
    a: dict[str, float],
    b: dict[str, float],
    *,
    resamples: int = 10_000,
    seed: int = 12345,
) -> dict[str, float]:
    """Compare per-query scores for systems A and B over their shared queries.

    ``a`` and ``b`` map qid -> metric value. Only queries present in both are used, since an unpaired
    query carries no information about the difference.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("no shared queries between the two runs")

    va = np.array([a[q] for q in shared], dtype=np.float64)
    vb = np.array([b[q] for q in shared], dtype=np.float64)
    diff = va - vb
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(resamples, len(shared)))
    means = diff[idx].mean(axis=1)

    # Two-sided p-value by the standard shift method: centre the bootstrap distribution on the null
    # (zero mean difference), then ask how often a resample is at least as extreme as what we saw.
    centred = means - observed
    p = float((np.abs(centred) >= abs(observed)).mean())

    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_queries": len(shared),
        "mean_a": float(va.mean()),
        "mean_b": float(vb.mean()),
        "diff": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p": p,
        "wins": int((diff > 1e-12).sum()),
        "losses": int((diff < -1e-12).sum()),
        "ties": int((np.abs(diff) <= 1e-12).sum()),
    }


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def format_comparisons(rows: list[tuple[str, str, dict]], metric: str) -> str:
    """Render a list of (name_a, name_b, result) as an aligned table."""
    width = max([len(a) for a, _, _ in rows] + [len(b) for _, b, _ in rows] + [8]) + 2
    out = [
        f"metric = {metric}",
        f"{'A':<{width}}{'B':<{width}}{'mean A':>9}{'mean B':>9}{'diff':>9}"
        f"{'95% CI':>22}{'p':>9}{'W/L/T':>16}",
        "-" * (2 * width + 74),
    ]
    for name_a, name_b, r in rows:
        ci = f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]"
        wlt = f"{r['wins']}/{r['losses']}/{r['ties']}"
        out.append(
            f"{name_a:<{width}}{name_b:<{width}}{r['mean_a']:>9.4f}{r['mean_b']:>9.4f}"
            f"{r['diff']:>+9.4f}{ci:>22}{r['p']:>9.4f}{wlt:>13} {stars(r['p'])}")
    out.append("-" * (2 * width + 74))
    out.append("* p<0.05  ** p<0.01  *** p<0.001   (paired bootstrap over queries)")
    return "\n".join(out)


if __name__ == "__main__":
    # Self-test on constructed data, so the implementation is checked rather than assumed.
    rng = np.random.default_rng(0)
    qids = [f"q{i}" for i in range(400)]

    # Identical systems: the difference must not be significant.
    base = {q: float(v) for q, v in zip(qids, rng.random(len(qids)))}
    same = paired_bootstrap(base, dict(base))
    assert same["diff"] == 0.0
    assert same["p"] > 0.9, same
    assert same["ties"] == len(qids)

    # A uniform +0.10 shift on every query is unmistakable.
    shifted = {q: v + 0.10 for q, v in base.items()}
    strong = paired_bootstrap(shifted, base)
    assert abs(strong["diff"] - 0.10) < 1e-9, strong
    assert strong["p"] < 0.001, strong
    assert strong["wins"] == len(qids)
    assert strong["ci_low"] > 0

    # Pure noise with mean zero should not reach significance.
    noisy = {q: v + float(d) for (q, v), d in zip(base.items(), rng.normal(0, 0.2, len(qids)))}
    weak = paired_bootstrap(noisy, base)
    assert weak["p"] > 0.05, weak

    # A tiny mean difference carried by a handful of queries stays non-significant, and the W/L/T
    # counts reveal why -- this is the case the test exists to catch.
    sparse = dict(base)
    for q in qids[:3]:
        sparse[q] = base[q] + 0.5
    r = paired_bootstrap(sparse, base)
    assert r["wins"] == 3 and r["losses"] == 0, r
    assert r["p"] > 0.05, r

    print(format_comparisons([
        ("identical", "base", same),
        ("shift+0.10", "base", strong),
        ("noise", "base", weak),
        ("3-query-spike", "base", r),
    ], "nDCG@10"))
    print("\nsignificance.py self-test PASSED")
