# CMGT-DINO Horti-M3 v1.0

## Proposed method

**CMGT-DINO = Cross-Modal Growth Trajectory Learning with Frozen DINOv3**

The final representation contains four research components:

1. **DINOv3 phenology-aware visual trajectory**
   - all RGB images are encoded once with a frozen DINOv3 backbone;
   - each plant-year is divided chronologically into early, middle and late growth stages;
   - stage embeddings and E→M, M→L, E→L visual changes are retained.

2. **Longitudinal phenotype trajectory**
   - stage mean, standard deviation, last observation and missingness;
   - early-to-middle, middle-to-late and early-to-late changes;
   - relative growth and whole-season slopes.

3. **Environmental exposure trajectory**
   - stage statistics for seven sensor variables;
   - VPD derived from temperature and relative humidity;
   - growing-degree-hour, high-temperature-hour, VPD exposure,
     light exposure proxy and CO2 exposure proxy.

4. **Cross-Modal Growth Concordance**
   - visual / phenotype trajectory agreement;
   - visual / environment trajectory agreement;
   - phenotype / environment trajectory agreement;
   - pairwise cosine, correlation, change-product, acceleration agreement
     and direction-consistency descriptors.

The final head is computationally light:
ExtraTrees, CatBoost, HistGradientBoosting and XGBoost are evaluated by inner OOF validation. A non-negative OOF blend is retained only if it reduces OOF RMSE relative to the strongest single head.

---

## Default visual backbone

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

DINOv3 ViT-S/16 is frozen. No end-to-end ViT training is performed.

A separate backbone ablation compares:

```text
EfficientNet-B0
ConvNeXt-Tiny
DINOv2-Small
DINOv3 ViT-S/16
DINOv3 ConvNeXt-Tiny
```

under the same plant-level trajectory protocol.

---

## Important DINOv3 dependency

DINOv3 is loaded through Hugging Face Transformers.

Before the first run:

```bash
pip install -r requirements.txt
python3 preflight_dinov3.py
```

If Hugging Face requests license/access acceptance, accept the official Meta DINOv3 model terms in the browser and authenticate locally:

```bash
huggingface-cli login
```

A downloaded local model directory can also be placed in `vision.backbones.dinov3_vits16.model_id`.

---

## Dataset configuration

The default Horti-M3 path is:

```yaml
data:
  root: /home/vaithees/PhD-Projects/Aasha_Christhuraj/HortiM3_RAMF_Complete/2023-2025 Tomato dataset
```

The package first tries to reuse the verified SCoRe-Fuse v1.3 three-year manifest:

```text
../SCoRe_Fuse_HortiM3_ThreeYear_v1_3/manifest_outputs/Multimodal_Manifest_2023_2025.csv
```

That manifest already contains:

```text
2023 + 2024 + 2025
207 plant-year targets
12,402 longitudinal target-aligned rows
```

If unavailable, the included validated v1.3 manifest builder is used.

---

## Run sequence

```bash
source ~/PhD-Projects/Aasha_Christhuraj/env_ramf/bin/activate

cd ~/PhD-Projects/Aasha_Christhuraj/CMGT_DINO_HortiM3_v1_0

pip install -r requirements.txt

python3 test_identity_harmonization.py
python3 prepare_manifest.py
python3 preflight_dinov3.py
python3 smoke_test.py

export MPLBACKEND=Agg
python3 run_cmgt_dino.py
```

The first DINOv3 run will encode approximately 4,811 unique RGB images.
Subsequent runs reuse:

```text
cmgt_cache/vision/dinov3_vits16/
```

---

## Backbone ablation

After the main run:

```bash
python3 run_backbone_ablation.py
```

The script creates separate caches for each backbone.

Output:

```text
Table_B1_Backbone_Ablation_All_Repeats.csv
Table_B2_Backbone_Ablation_Summary.csv
```

## Main baselines

The main run includes:

```text
DINOv3 Static Image ExtraTrees
Phenotype-only ExtraTrees
Environment-only ExtraTrees
Static Early Fusion ExtraTrees
Static Early Fusion XGBoost
Static Early Fusion CatBoost
Static Early Fusion HistGB
Trajectory Fusion ExtraTrees
CMGT-DINO
```

This structure separates the effect of:

- stronger visual foundation features;
- temporal/phenology trajectory representation;
- explicit cross-modal growth concordance;
- final adaptive lightweight regression head.

---

## Ablation

Generated variants:

```text
Full CMGT-DINO
Without Concordance
Without Environment Trajectory
Without Phenotype Trajectory
Image Trajectory Only
```

The key novelty test is:

```text
Full CMGT-DINO
vs
Without Concordance
```

---

## Validation protocol

### Repeated three-year held-out evaluation

Default:

```text
20 repeats
20% test fraction
year-stratified test selection
```

All three seasons remain represented in each random held-out test set.

### Leave-one-year-out

```text
Train 2024 + 2025 → Test 2023
Train 2023 + 2025 → Test 2024
Train 2023 + 2024 → Test 2025
```

This directly tests cross-season domain generalization.

---

## Metrics

```text
RMSE
MAE
R²
NRMSE
sMAPE
```

MAPE is intentionally not used because Horti-M3 contains zero-yield values.

---

## Statistics

Generated:

```text
Friedman omnibus test
Paired Wilcoxon signed-rank tests
Holm correction
Paired Cohen's dz
Bootstrap 95% CI of paired RMSE difference
```

---

## Explainability

CMGT-DINO XAI is designed to remain agronomically interpretable.

Generated:

```text
XAI_Block_Permutation_Importance.csv
XAI_Block_Permutation_Importance.png
XAI_Concordance_Feature_Importance.csv
XAI_Concordance_Feature_Importance.png
```

The main block-level explanation reports the held-out contribution of:

```text
image trajectory
phenotype trajectory
environment trajectory
cross-modal concordance
acquisition/reliability metadata
```

The concordance explanation uses named features such as:

```text
image_phenotype__cosine
image_environment__acceleration_agreement
phenotype_environment__correlation
tri_modal_cosine_mean
```

instead of only uninterpretable PCA labels.

---

## Main outputs

```text
cmgt_outputs/tables/Table_1_Input_Audit.csv
cmgt_outputs/tables/Table_2_Plant_Year_Sample_Audit.csv
cmgt_outputs/tables/Table_3_Feature_Block_Audit.csv
cmgt_outputs/tables/Table_4_All_Repeated_Metrics.csv
cmgt_outputs/tables/Table_5_Model_Summary.csv
cmgt_outputs/tables/Table_6_Runtime.csv
cmgt_outputs/tables/Table_7_Ablation.csv
cmgt_outputs/tables/Table_8_Friedman.csv
cmgt_outputs/tables/Table_9_Wilcoxon_Holm_EffectSize.csv
cmgt_outputs/tables/Table_10_Leave_One_Year_Out.csv
cmgt_outputs/tables/Prediction_Detail.csv
cmgt_outputs/tables/LOYO_Prediction_Detail.csv
cmgt_outputs/CMGT_DINO_All_Results.xlsx
```

All figures are saved at 600 DPI.

---

## Scientific limitation

No implementation can guarantee that CMGT-DINO will beat every baseline on unseen data before execution.

The implementation is deliberately structured to give the proposed representation a fair and strong test:

- DINOv3 instead of EfficientNet-B0;
- chronological phenology retained instead of whole-season averaging;
- explicit change descriptors;
- cross-modal trajectory concordance;
- lightweight OOF-selected head;
- year-stratified repeated validation;
- leave-one-year-out validation.

The final superiority claim must be based on `Table_5_Model_Summary.csv`,
`Table_9_Wilcoxon_Holm_EffectSize.csv`, and `Table_10_Leave_One_Year_Out.csv`.
