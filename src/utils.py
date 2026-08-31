from __future__ import annotations

from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_output_dirs(base):
    base = Path(base)
    dirs = {
        "base": base,
        "tables": base / "tables",
        "figures": base / "figures",
        "models": base / "models",
        "xai": base / "xai",
        "logs": base / "logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def linear_slope(values):
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.nan
    x = np.arange(len(v), dtype=float)[ok]
    y = v[ok]
    x = x - x.mean()
    den = np.sum(x * x)
    if den <= 0:
        return 0.0
    return float(np.sum(x * (y - y.mean())) / den)


def smape(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    den = np.abs(y) + np.abs(p)
    return float(np.mean(200.0 * np.abs(p - y) / np.maximum(den, 1e-8)))


def regression_metrics(y, p):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y = np.asarray(y, float)
    p = np.asarray(p, float)

    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(mean_absolute_error(y, p))
    r2 = float(r2_score(y, p)) if len(y) > 1 else np.nan
    rng = float(np.nanmax(y) - np.nanmin(y)) if len(y) else np.nan
    nrmse = rmse / max(rng, 1e-8)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "NRMSE": nrmse,
        "sMAPE": smape(y, p),
    }


def safe_corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return 0.0
    aa = a[ok]
    bb = b[ok]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def cosine(a, b):
    a = np.nan_to_num(np.asarray(a, float), nan=0.0)
    b = np.nan_to_num(np.asarray(b, float), nan=0.0)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def export_excel_safe(path, sheets):
    """
    Excel export that cannot fail with 'At least one sheet must be visible'.
    A README sheet is written first and every subsequent sheet is isolated.
    """
    path = Path(path)
    errors = []
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({
            "CMGT-DINO": [
                "Generated result workbook",
                "CSV files in tables/ remain the canonical machine-readable outputs."
            ]
        }).to_excel(writer, sheet_name="README", index=False)

        for name, df in sheets.items():
            try:
                if df is None:
                    continue
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                safe = str(name)[:31] or "Sheet"
                df.to_excel(writer, sheet_name=safe, index=False)
            except Exception as e:
                errors.append((name, repr(e)))

    return errors
