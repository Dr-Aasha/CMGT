from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import models

from .image_paths import resolved_image_paths_for_manifest


class HFVisionEncoder:
    def __init__(self, model_id, device="auto", disable_cudnn=True, local_files_only=False):
        from transformers import AutoImageProcessor, AutoModel

        if disable_cudnn:
            torch.backends.cudnn.enabled = False

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_id = model_id
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            local_files_only=bool(local_files_only),
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            local_files_only=bool(local_files_only),
        ).to(device).eval()

        for p in self.model.parameters():
            p.requires_grad = False

        print(
            f"[VISION] Loaded {model_id} on {device}; "
            f"cuDNN enabled={torch.backends.cudnn.enabled}"
        )

    @torch.inference_mode()
    def encode_batch(self, paths):
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError(
                    f"{self.model_id} returned neither pooler_output nor last_hidden_state."
                )
            # CLS token for ViT-style models.
            pooled = hidden[:, 0]

        return pooled.detach().float().cpu().numpy().astype(np.float32)


class TorchvisionEncoder:
    def __init__(self, name, device="auto", disable_cudnn=True):
        if disable_cudnn:
            torch.backends.cudnn.enabled = False

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.name = name

        if name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            net = models.efficientnet_b0(weights=weights)
            self.features = net.features.to(device).eval()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.transform = weights.transforms()

        elif name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT
            net = models.convnext_tiny(weights=weights)
            self.features = net.features.to(device).eval()
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.transform = weights.transforms()

        else:
            raise ValueError(name)

        for p in self.features.parameters():
            p.requires_grad = False

        print(
            f"[VISION] Loaded torchvision {name} on {device}; "
            f"cuDNN enabled={torch.backends.cudnn.enabled}"
        )

    @torch.inference_mode()
    def encode_batch(self, paths):
        xs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            xs.append(self.transform(img))
        x = torch.stack(xs).to(self.device)
        z = self.pool(self.features(x)).flatten(1)
        return z.detach().float().cpu().numpy().astype(np.float32)


def make_encoder(backbone_name, cfg):
    spec = cfg["vision"]["backbones"][backbone_name]
    kind = spec["type"]
    device = cfg["vision"].get("device", "auto")
    disable_cudnn = cfg["vision"].get("disable_cudnn", True)

    if kind == "huggingface":
        return HFVisionEncoder(
            spec["model_id"],
            device=device,
            disable_cudnn=disable_cudnn,
            local_files_only=cfg["vision"].get("local_files_only", False),
        )

    if kind == "torchvision":
        return TorchvisionEncoder(
            spec["model_id"],
            device=device,
            disable_cudnn=disable_cudnn,
        )

    raise ValueError(f"Unknown backbone type: {kind}")


def _signature(backbone_name, paths):
    text = backbone_name + "\n" + "\n".join(sorted(paths))
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def build_embedding_cache(manifest, cfg, backbone_name=None):
    if backbone_name is None:
        backbone_name = cfg["vision"]["proposed_backbone"]

    cache_root = Path(cfg["data"]["cache_dir"]) / "vision" / backbone_name
    cache_root.mkdir(parents=True, exist_ok=True)

    idx_path = cache_root / "embedding_index.csv"
    npz_path = cache_root / "embeddings.npz"
    sig_path = cache_root / "signature.txt"
    fail_path = cache_root / "failures.csv"

    paths, declared, _ = resolved_image_paths_for_manifest(
        manifest,
        cfg["data"]["root"],
    )

    sig = _signature(backbone_name, paths)

    print(f"[VISION] Declared image references: {declared}")
    print(f"[VISION] Unique resolved images: {len(paths)}")
    print(f"[VISION] Backbone: {backbone_name}")

    if idx_path.exists() and npz_path.exists() and sig_path.exists():
        try:
            old_sig = sig_path.read_text(encoding="utf-8").strip()
            meta = pd.read_csv(idx_path)
            arr = np.load(npz_path)["embeddings"]
            if old_sig == sig and len(meta) == len(arr) and len(arr):
                print(f"[VISION] Reusing cache: {arr.shape}")
                return arr, meta
        except Exception:
            pass

    # Partial reuse from current cache even if manifest expanded.
    old_map = {}
    old_arr = None
    if idx_path.exists() and npz_path.exists():
        try:
            old_meta = pd.read_csv(idx_path)
            old_arr = np.load(npz_path)["embeddings"]
            for r in old_meta.itertuples():
                old_map[str(r.image_path)] = int(r.index)
        except Exception:
            old_map = {}
            old_arr = None

    reusable = [p for p in paths if p in old_map]
    new_paths = [p for p in paths if p not in old_map]

    print(f"[VISION] Reusable cached embeddings: {len(reusable)}")
    print(f"[VISION] New embeddings required: {len(new_paths)}")

    encoder = None
    if new_paths:
        encoder = make_encoder(backbone_name, cfg)

    batch_size = int(cfg["vision"].get("batch_size", 12))

    new_vectors = {}
    failures = []

    for start in tqdm(
        range(0, len(new_paths), batch_size),
        desc=f"{backbone_name} embeddings",
    ):
        batch = new_paths[start:start + batch_size]
        try:
            z = encoder.encode_batch(batch)
            for p, vec in zip(batch, z):
                new_vectors[p] = vec
        except RuntimeError as e:
            # Safe OOM / batch fallback to single-image inference.
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()

            for p in batch:
                try:
                    vec = encoder.encode_batch([p])[0]
                    new_vectors[p] = vec
                except Exception as ee:
                    failures.append({"image_path": p, "error": repr(ee)})
        except Exception as e:
            for p in batch:
                try:
                    vec = encoder.encode_batch([p])[0]
                    new_vectors[p] = vec
                except Exception as ee:
                    failures.append({"image_path": p, "error": repr(ee)})

    vectors = []
    rows = []

    for p in paths:
        if p in new_vectors:
            vec = new_vectors[p]
        elif p in old_map and old_arr is not None:
            vec = old_arr[old_map[p]]
        else:
            continue

        rows.append({"image_path": p, "index": len(vectors)})
        vectors.append(np.asarray(vec, np.float32))

    if not vectors:
        raise RuntimeError(f"No image embeddings generated for {backbone_name}.")

    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        raise RuntimeError(f"Embedding dimension mismatch for {backbone_name}: {dims}")

    arr = np.stack(vectors).astype(np.float32)
    meta = pd.DataFrame(rows)

    np.savez_compressed(npz_path, embeddings=arr)
    meta.to_csv(idx_path, index=False)
    pd.DataFrame(failures).to_csv(fail_path, index=False)
    sig_path.write_text(sig, encoding="utf-8")

    print(
        f"[VISION] Cache completed: {arr.shape}; "
        f"new={len(new_vectors)}; failures={len(failures)}"
    )

    return arr, meta
