from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import regression_metrics


PROPOSED_BLOCKS = [
    "image_trajectory",
    "phenotype_trajectory",
    "environment_trajectory",
    "concordance",
    "meta_reliability",
]


def _predict_bundle(bundle, data, idx):
    pp = bundle["preprocessor"]
    model = bundle["model"]
    X = pp.proposed_matrix(data, idx)
    return model.predict(X)


def block_permutation_importance(bundle, data, cfg, outdirs):
    test_idx = np.asarray(bundle["test_idx"], dtype=int)
    y = data["y"][test_idx]

    base_pred = _predict_bundle(bundle, data, test_idx)
    base_rmse = regression_metrics(y, base_pred)["RMSE"]

    rng = np.random.default_rng(cfg["seed"])
    repeats = int(cfg["xai"].get("block_permutation_repeats", 20))

    rows = []

    for block in PROPOSED_BLOCKS:
        diffs = []

        for _ in range(repeats):
            original = data[block][test_idx].copy()
            shuffled = original[rng.permutation(len(original))]

            # Temporary shallow copy with test block replaced.
            temp = dict(data)
            arr = data[block].copy()
            arr[test_idx] = shuffled
            temp[block] = arr

            p = _predict_bundle(bundle, temp, test_idx)
            diffs.append(regression_metrics(y, p)["RMSE"] - base_rmse)

        rows.append({
            "block": block,
            "baseline_RMSE": base_rmse,
            "RMSE_increase_mean": float(np.mean(diffs)),
            "RMSE_increase_std": float(np.std(diffs)),
        })

    df = pd.DataFrame(rows).sort_values("RMSE_increase_mean", ascending=False)
    df.to_csv(outdirs["xai"] / "XAI_Block_Permutation_Importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    q = df.sort_values("RMSE_increase_mean")
    plt.barh(q["block"], q["RMSE_increase_mean"])
    plt.xlabel("RMSE increase after block permutation", fontsize=13)
    plt.title("CMGT-DINO modality/block importance", fontsize=15)
    plt.tight_layout()
    plt.savefig(
        outdirs["xai"] / "XAI_Block_Permutation_Importance.png",
        dpi=int(cfg["output"].get("dpi", 600)),
        bbox_inches="tight",
    )
    plt.close("all")

    return df


def concordance_feature_importance(bundle, data, cfg, outdirs):
    test_idx = np.asarray(bundle["test_idx"], dtype=int)
    y = data["y"][test_idx]
    pp = bundle["preprocessor"]
    model = bundle["model"]

    base = model.predict(pp.proposed_matrix(data, test_idx))
    base_rmse = regression_metrics(y, base)["RMSE"]

    names = data["feature_names"]["concordance"]
    Xraw = data["concordance"][test_idx].copy()

    rng = np.random.default_rng(cfg["seed"] + 101)
    repeats = int(cfg["xai"].get("engineered_feature_permutation_repeats", 5))

    rows = []
    for j, name in enumerate(names):
        diffs = []
        for _ in range(repeats):
            temp = dict(data)
            arr = data["concordance"].copy()
            changed = Xraw.copy()
            changed[:, j] = changed[rng.permutation(len(changed)), j]
            arr[test_idx] = changed
            temp["concordance"] = arr

            p = model.predict(pp.proposed_matrix(temp, test_idx))
            diffs.append(regression_metrics(y, p)["RMSE"] - base_rmse)

        rows.append({
            "feature": name,
            "RMSE_increase_mean": float(np.mean(diffs)),
            "RMSE_increase_std": float(np.std(diffs)),
        })

    df = pd.DataFrame(rows).sort_values("RMSE_increase_mean", ascending=False)
    df.to_csv(outdirs["xai"] / "XAI_Concordance_Feature_Importance.csv", index=False)

    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["RMSE_increase_mean"])
    plt.xlabel("RMSE increase after feature permutation", fontsize=13)
    plt.title("Growth-concordance feature importance", fontsize=15)
    plt.tight_layout()
    plt.savefig(
        outdirs["xai"] / "XAI_Concordance_Feature_Importance.png",
        dpi=int(cfg["output"].get("dpi", 600)),
        bbox_inches="tight",
    )
    plt.close("all")

    return df
