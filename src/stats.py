from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def holm_adjust(pvals):
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.zeros(m)
    running = 0.0

    for rank, idx in enumerate(order):
        value = (m - rank) * pvals[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)

    return adjusted


def cohen_dz(d):
    d = np.asarray(d, float)
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 0 else np.nan


def bootstrap_ci(d, n=2000, seed=42):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = [
        np.mean(rng.choice(d, size=len(d), replace=True))
        for _ in range(int(n))
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def statistical_tables(metrics_df, cfg):
    pivot = metrics_df.pivot(index="repeat", columns="model", values="RMSE").dropna(axis=1)

    if pivot.shape[1] >= 3:
        stat, p = friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
        friedman = pd.DataFrame([{
            "metric": "RMSE",
            "friedman_chi2": stat,
            "p_value": p,
            "n_models": pivot.shape[1],
            "n_repeats": pivot.shape[0],
        }])
    else:
        friedman = pd.DataFrame()

    rows = []
    if "CMGT-DINO" in pivot.columns:
        raw_p = []
        tmp = []

        proposed = pivot["CMGT-DINO"].to_numpy()

        for baseline in pivot.columns:
            if baseline == "CMGT-DINO":
                continue

            d = proposed - pivot[baseline].to_numpy()

            try:
                w, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            except Exception:
                w, p = np.nan, 1.0

            lo, hi = bootstrap_ci(
                d,
                cfg["experiment"].get("bootstrap_iterations", 2000),
                cfg["seed"],
            )

            tmp.append({
                "baseline": baseline,
                "CMGT_minus_baseline_RMSE": float(np.mean(d)),
                "wilcoxon_W": w,
                "p_value": p,
                "cohen_dz": cohen_dz(d),
                "bootstrap95_low": lo,
                "bootstrap95_high": hi,
            })
            raw_p.append(p)

        adjusted = holm_adjust(raw_p)

        for row, hp in zip(tmp, adjusted):
            row["holm_p"] = float(hp)
            row["significant_0.05"] = bool(hp < 0.05)
            rows.append(row)

    return friedman, pd.DataFrame(rows)
