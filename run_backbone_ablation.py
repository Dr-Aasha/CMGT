import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
import shutil
import pandas as pd

from src.utils import load_config
from src.manifest import build_three_year_manifest
from src.backbone_ablation import evaluate_backbones


def ensure_manifest(cfg):
    target = Path(cfg["data"]["manifest_csv"])
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        for candidate in cfg["data"].get("external_manifest_candidates", []):
            p = Path(candidate)
            if p.exists():
                shutil.copy2(p, target)
                break

    if not target.exists():
        build_three_year_manifest(cfg)

    return pd.read_csv(target)


def main():
    cfg = load_config("config.yaml")
    manifest = ensure_manifest(cfg)

    out_dir = Path(cfg["data"]["output_dir"])
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    results = evaluate_backbones(manifest, cfg)

    results.to_csv(
        tables / "Table_B1_Backbone_Ablation_All_Repeats.csv",
        index=False,
    )

    summary = (
        results.groupby("backbone")
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            R2_mean=("R2", "mean"),
        )
        .reset_index()
        .sort_values("RMSE_mean")
    )

    summary.to_csv(
        tables / "Table_B2_Backbone_Ablation_Summary.csv",
        index=False,
    )

    print("\nBackbone ablation completed.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
