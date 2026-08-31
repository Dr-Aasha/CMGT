import numpy as np
from src.modeling import OOFBlendRegressor, CMGTPreprocessor
from src.utils import regression_metrics


def main():
    rng = np.random.default_rng(42)
    n = 120

    latent = rng.normal(size=(n, 5))
    y = 3.0 * latent[:, 0] - 2.0 * latent[:, 1] + rng.normal(0, 0.4, n)

    data = {
        "image_trajectory": rng.normal(size=(n, 80)).astype(np.float32),
        "phenotype_trajectory": np.column_stack([
            latent[:, 0],
            latent[:, 1],
            rng.normal(size=(n, 18)),
        ]).astype(np.float32),
        "environment_trajectory": np.column_stack([
            latent[:, 2],
            rng.normal(size=(n, 19)),
        ]).astype(np.float32),
        "concordance": np.column_stack([
            latent[:, 0] * latent[:, 1],
            rng.normal(size=(n, 9)),
        ]).astype(np.float32),
        "meta_reliability": rng.normal(size=(n, 6)).astype(np.float32),
        "static_image": rng.normal(size=(n, 40)).astype(np.float32),
        "static_phenotype": rng.normal(size=(n, 16)).astype(np.float32),
        "static_environment": rng.normal(size=(n, 12)).astype(np.float32),
        "feature_names": {},
        "y": y,
    }

    for k in [
        "image_trajectory",
        "phenotype_trajectory",
        "environment_trajectory",
        "concordance",
        "meta_reliability",
        "static_image",
        "static_phenotype",
        "static_environment",
    ]:
        data["feature_names"][k] = [f"{k}_{j}" for j in range(data[k].shape[1])]

    cfg = {
        "features": {
            "image_pca_components": 16,
            "phenotype_pca_components": 12,
            "environment_pca_components": 10,
            "static_image_pca_components": 12,
            "static_phenotype_pca_components": 10,
            "static_environment_pca_components": 8,
        }
    }

    tr = np.arange(90)
    te = np.arange(90, 120)

    pp = CMGTPreprocessor(cfg).fit(data, tr)
    Xtr = pp.proposed_matrix(data, tr)
    Xte = pp.proposed_matrix(data, te)

    model = OOFBlendRegressor(
        ["extratrees", "histgb"],
        folds=4,
        seed=42,
        use_blend=True,
    ).fit(Xtr, y[tr])

    pred = model.predict(Xte)
    assert np.isfinite(pred).all()
    assert pred.shape == (30,)

    print("CMGT-DINO smoke test PASSED")
    print("Transformed shape:", Xtr.shape)
    print("Model audit:", model.audit())
    print("Synthetic metrics:", regression_metrics(y[te], pred))


if __name__ == "__main__":
    main()
