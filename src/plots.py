from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt


def _save(path, dpi):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")


def make_plots(summary, ablation, predictions, loyo, outdirs, cfg):
    dpi = int(cfg["output"].get("dpi", 600))

    # Model comparison.
    s = summary.sort_values("RMSE_mean", ascending=True)

    plt.figure(figsize=(11, 7))
    plt.barh(s["model"], s["RMSE_mean"], xerr=s["RMSE_std"])
    plt.xlabel("RMSE", fontsize=14)
    plt.ylabel("Model", fontsize=14)
    plt.title("CMGT-DINO clean three-year comparison", fontsize=16)
    _save(outdirs["figures"] / "Figure_1_Model_RMSE.png", dpi)

    # Ablation.
    a = (
        ablation.groupby("variant")["RMSE"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean")
    )

    plt.figure(figsize=(10, 6))
    plt.barh(a["variant"], a["mean"], xerr=a["std"])
    plt.xlabel("RMSE", fontsize=14)
    plt.ylabel("Variant", fontsize=14)
    plt.title("CMGT-DINO ablation", fontsize=16)
    _save(outdirs["figures"] / "Figure_2_Ablation.png", dpi)

    # Observed vs predicted repeat 0.
    p = predictions[
        (predictions["repeat"] == 0)
        & (predictions["model"] == "CMGT-DINO")
    ]

    if len(p):
        plt.figure(figsize=(7, 7))
        plt.scatter(p["y_true"], p["y_pred"], alpha=0.75)
        lo = min(p["y_true"].min(), p["y_pred"].min())
        hi = max(p["y_true"].max(), p["y_pred"].max())
        plt.plot([lo, hi], [lo, hi], "--")
        plt.xlabel("Observed yield", fontsize=14)
        plt.ylabel("Predicted yield", fontsize=14)
        plt.title("CMGT-DINO observed vs predicted", fontsize=16)
        _save(outdirs["figures"] / "Figure_3_Observed_vs_Predicted.png", dpi)

    if loyo is not None and len(loyo):
        q = loyo[loyo["model"] == "CMGT-DINO"].sort_values("heldout_year")
        plt.figure(figsize=(8, 6))
        plt.bar(q["heldout_year"].astype(str), q["RMSE"])
        plt.xlabel("Held-out year", fontsize=14)
        plt.ylabel("RMSE", fontsize=14)
        plt.title("CMGT-DINO leave-one-year-out generalization", fontsize=16)
        _save(outdirs["figures"] / "Figure_4_LOYO.png", dpi)
