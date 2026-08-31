from __future__ import annotations

import numpy as np
import pandas as pd

from .backbones import build_embedding_cache
from .trajectory_features import build_cmgt_samples
from .sensors import SensorRepository
from .modeling import CMGTPreprocessor, fit_extratrees
from .evaluate import year_stratified_split
from .utils import regression_metrics


def evaluate_backbones(manifest, cfg):
    rows = []

    sensor_repo = SensorRepository(
        cfg["data"]["root"],
        cfg["data"]["sensor_features"],
    )

    repeats = int(cfg["backbone_ablation"].get("repeats", 10))

    for backbone in cfg["backbone_ablation"]["enabled_backbones"]:
        print(f"\n[BACKBONE] ===== {backbone} =====")

        emb, meta = build_embedding_cache(
            manifest,
            cfg,
            backbone_name=backbone,
        )

        data = build_cmgt_samples(
            manifest,
            emb,
            meta,
            sensor_repo,
            cfg,
        )

        for rep in range(repeats):
            seed = cfg["seed"] + 10000 + rep

            tr, te = year_stratified_split(
                data["meta"],
                cfg["experiment"]["test_fraction"],
                seed,
            )

            pp = CMGTPreprocessor(cfg).fit(data, tr)

            # Identical image-trajectory ExtraTrees head for fair backbone comparison.
            Xtr = pp.transform_block(data, tr, "image_trajectory")
            Xte = pp.transform_block(data, te, "image_trajectory")

            model = fit_extratrees(Xtr, data["y"][tr], seed)
            p = model.predict(Xte)

            rows.append({
                "backbone": backbone,
                "repeat": rep,
                "embedding_dim": int(emb.shape[1]),
                **regression_metrics(data["y"][te], p),
            })

            print(
                f"[BACKBONE] {backbone} repeat {rep+1}/{repeats}: "
                f"RMSE={rows[-1]['RMSE']:.4f}"
            )

    return pd.DataFrame(rows)
