from pathlib import Path
import numpy as np
from PIL import Image

from src.utils import load_config
from src.backbones import make_encoder


def main():
    cfg = load_config("config.yaml")
    name = cfg["vision"]["proposed_backbone"]

    print("Backbone:", name)
    print("Model:", cfg["vision"]["backbones"][name])

    encoder = make_encoder(name, cfg)

    tmp = Path("cmgt_cache") / "_preflight_image.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    image = (rng.random((224, 224, 3)) * 255).astype(np.uint8)
    Image.fromarray(image).save(tmp)

    z = encoder.encode_batch([str(tmp)])
    print("DINOv3 preflight SUCCESS")
    print("Embedding shape:", z.shape)

    try:
        tmp.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    main()
