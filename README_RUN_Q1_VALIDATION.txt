CMGT-DINOv3 — Q1 Journal Validation Additions
==============================================

Purpose
-------
This package adds the two experiments recommended before submission:

1. Image-overlap / leakage audit
2. DINOv3 vs DINOv2 vs EfficientNet-B0 vs ConvNeXt-Tiny backbone ablation

Copy all three Python files into:
    ~/PhD-Projects/Aasha_Christhuraj/CMGT_DINO_HortiM3_v1_0/

Activate:
    source ~/PhD-Projects/Aasha_Christhuraj/env_ramf/bin/activate
    cd ~/PhD-Projects/Aasha_Christhuraj/CMGT_DINO_HortiM3_v1_0
    export MPLBACKEND=Agg

STEP 1 — Leakage audit + leakage-safe re-evaluation
----------------------------------------------------
python3 q1_image_leakage_audit.py --repeats 20

Important outputs:
cmgt_outputs/q1_validation/leakage/Table_L1_Global_Image_Sharing_Summary.csv
cmgt_outputs/q1_validation/leakage/Table_L3_Original_Split_Image_Overlap.csv
cmgt_outputs/q1_validation/leakage/Table_L5b_LeakageSafe_Split_Image_Overlap.csv
cmgt_outputs/q1_validation/leakage/Leakage_Audit_Interpretation.txt
cmgt_outputs/q1_validation/leakage_safe_evaluation/Table_L7_LeakageSafe_Model_Summary.csv

Interpretation:
- If Table_L3 has shared_image_paths = 0 for all 20 repeats, the original splits
  show no identical raw-image path leakage under this audit.
- If Table_L3 contains values > 0, use the leakage-safe results in the paper.
- Table_L5b should be zero for every repeat by construction.

Audit only, without rerunning models:
python3 q1_image_leakage_audit.py --repeats 20 --no-safe-rerun

STEP 2 — Backbone ablation
---------------------------
Recommended: use image-cohort-safe splits.

python3 q1_backbone_ablation.py --repeats 20 --split-mode safe

Backbones:
- EfficientNet-B0
- ConvNeXt-Tiny
- DINOv2-Small
- DINOv3 ViT-S/16

The experiment evaluates each backbone under three identical protocols:
1. Static Image ExtraTrees
2. Image Trajectory ExtraTrees
3. Full CMGT

Important outputs:
cmgt_outputs/q1_validation/backbone_ablation/Table_B2_Backbone_Ablation_Summary.csv
cmgt_outputs/q1_validation/backbone_ablation/Table_B3_Backbone_Friedman.csv
cmgt_outputs/q1_validation/backbone_ablation/Table_B4_DINOv3_vs_Backbones_Paired_Statistics.csv
cmgt_outputs/q1_validation/backbone_ablation/Figure_B1_Full_CMGT_Backbone_RMSE.png
cmgt_outputs/q1_validation/backbone_ablation/Figure_B2_Backbone_Protocols_RMSE.png

Why three protocols?
--------------------
Static Image ExtraTrees isolates the frozen visual representation.
Image Trajectory ExtraTrees tests whether the backbone improves temporal visual modeling.
Full CMGT tests the final end-to-end proposed representation while keeping the rest of the
pipeline identical.

Important computational note
----------------------------
The existing DINOv3 cache is reused automatically.
EfficientNet-B0, ConvNeXt-Tiny and DINOv2-Small are encoded once and cached under:
    cmgt_cache/vision/<backbone>/

DINOv3 gated access must already be authorized.
