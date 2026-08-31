from __future__ import annotations

from pathlib import Path
import os
import re
import numpy as np
import pandas as pd


ALIASES = {
    "Voltage": ["Voltage", "电压"],
    "Air Temperature": ["Air Temperature", "Temperature", "Air Temp", "空气温度", "气温"],
    "Relative Humidity": ["Relative Humidity", "Humidity", "相对湿度", "湿度"],
    "Light Intensity": ["Light Intensity", "Light", "Illumination", "光照强度", "光照"],
    "CO2": ["CO2", "CO₂", "二氧化碳"],
    "Soil Moisture": ["Soil Moisture", "Soil Humidity", "土壤湿度", "基质湿度"],
    "Soil Temperature": ["Soil Temperature", "土壤温度", "基质温度"],
}


def norm(s):
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(s).lower())


def find_col(cols, names):
    nmap = {norm(c): c for c in cols}
    for n in names:
        if norm(n) in nmap:
            return nmap[norm(n)]
    for c in cols:
        nc = norm(c)
        for n in names:
            nn = norm(n)
            if nn and (nn in nc or nc in nn):
                return c
    return None


def read_csv_flexible(path):
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return None


def read_sensor_tables(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = read_csv_flexible(path)
        return [("csv", df)] if df is not None and not df.empty else []
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            book = pd.read_excel(path, sheet_name=None)
            return [(str(k), v) for k, v in book.items() if v is not None and not v.empty]
        except Exception:
            return []
    return []


def extract_sensor_frame(df, requested):
    cmap = {
        f: find_col(df.columns, ALIASES.get(f, [f]))
        for f in requested
    }
    if sum(c is not None for c in cmap.values()) < 2:
        return None

    out = pd.DataFrame(index=np.arange(len(df)))
    for f in requested:
        if cmap[f] is not None:
            out[f] = pd.to_numeric(df[cmap[f]], errors="coerce")
        else:
            out[f] = np.nan
    return out


def infer_year(path):
    m = re.search(r"(20\d{2})", str(path))
    return int(m.group(1)) if m else None


def infer_date(path, sheet, df):
    year = infer_year(path)
    s = str(sheet)

    m = re.search(r"(20\d{2})[-_/]?(\d{1,2})[-_/]?(\d{1,2})", s)
    if m:
        try:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3))).normalize()
        except Exception:
            pass

    m = re.fullmatch(r"(\d{2})(\d{2})", s.strip())
    if m and year:
        try:
            return pd.Timestamp(year, int(m.group(1)), int(m.group(2))).normalize()
        except Exception:
            pass

    for c in list(df.columns)[:3]:
        q = pd.to_datetime(df[c], errors="coerce")
        if q.notna().any():
            return q.dropna().iloc[0].normalize()

    # Filename fallback.
    stem = Path(path).stem
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", stem)
    if m:
        try:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3))).normalize()
        except Exception:
            pass

    return pd.NaT


def resolve_sensor_path(raw_path, source_file, root):
    if raw_path is None:
        return None
    s = str(raw_path).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    s = s.replace("\\", os.sep).replace("/", os.sep)
    p = Path(s)

    if p.is_absolute() and p.exists():
        return str(p.resolve())

    direct = Path(root) / p
    if direct.exists():
        return str(direct.resolve())

    if source_file:
        src = Path(source_file)
        for parent in list(src.parents)[:9]:
            cand = parent / p
            try:
                cand = cand.resolve()
            except Exception:
                pass
            if cand.exists():
                return str(cand)

    return None


class SensorRepository:
    def __init__(self, root, requested):
        self.root = Path(root)
        self.requested = list(requested)
        self.file_cache = {}
        self.date_store = {}
        self._build_date_store()

    def load_path(self, path):
        path = str(Path(path).resolve())
        if path in self.file_cache:
            return self.file_cache[path]

        frames = []
        for _, df in read_sensor_tables(path):
            x = extract_sensor_frame(df, self.requested)
            if x is not None:
                frames.append(x)

        out = pd.concat(frames, ignore_index=True) if frames else None
        self.file_cache[path] = out
        return out

    def _build_date_store(self):
        candidates = []
        for p in self.root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
                continue
            low = str(p).lower()
            if "sensor" in low or "传感" in low:
                candidates.append(p)

        for p in candidates:
            for sheet, df in read_sensor_tables(p):
                x = extract_sensor_frame(df, self.requested)
                if x is None:
                    continue
                d = infer_date(p, sheet, df)
                if pd.isna(d):
                    continue
                self.date_store.setdefault(pd.Timestamp(d).normalize(), []).append(x)

        for d, frames in list(self.date_store.items()):
            self.date_store[d] = pd.concat(frames, ignore_index=True)

        print(f"[SENSOR] Date-level fallback days: {len(self.date_store)}")

    def frames_for_rows(self, frame):
        direct_paths = []

        if "sensor_raw" in frame.columns:
            for r in frame.itertuples():
                p = resolve_sensor_path(
                    getattr(r, "sensor_raw", None),
                    getattr(r, "manual_csv", None),
                    self.root,
                )
                if p:
                    direct_paths.append(p)

        direct_paths = list(dict.fromkeys(direct_paths))
        direct_frames = []
        for p in direct_paths:
            x = self.load_path(p)
            if x is not None:
                direct_frames.append(x)

        if direct_frames:
            return direct_frames, "direct", len(direct_paths)

        fallback = []
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna().dt.normalize().unique()
        for d in dates:
            x = self.date_store.get(pd.Timestamp(d).normalize())
            if x is not None:
                fallback.append(x)

        return fallback, "date_fallback", 0
