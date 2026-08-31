CMGT-DINOv3 — FINAL LOCO VALIDATION
===================================

Files
-----
q1_loco_validation.py

Requirement
-----------
Keep q1_validation_common.py from the earlier Q1 validation package in the
same CMGT project directory.

Copy q1_loco_validation.py into:
~/PhD-Projects/Aasha_Christhuraj/CMGT_DINO_HortiM3_v1_0/

Activate:
source ~/PhD-Projects/Aasha_Christhuraj/env_ramf/bin/activate

Move:
cd ~/PhD-Projects/Aasha_Christhuraj/CMGT_DINO_HortiM3_v1_0

Set plotting backend:
export MPLBACKEND=Agg

RECOMMENDED FULL RUN
--------------------
python3 q1_loco_validation.py --with-backbones

This runs:
1. 18-fold Leave-One-Image-Cohort-Out audit
2. Full CMGT-DINOv3 model comparison
3. CMGT ablation
4. Paired statistics
5. Per-year pooled analysis
6. DINOv3 vs DINOv2 vs ConvNeXt vs EfficientNet under identical LOCO folds

If the full backbone comparison will be run later:
python3 q1_loco_validation.py

Then later:
python3 q1_loco_validation.py --backbones-only

MOST IMPORTANT OUTPUTS
----------------------
cmgt_outputs/q1_validation/loco/Table_C1_LOCO_Fold_Audit.csv
cmgt_outputs/q1_validation/loco/Table_C4_LOCO_Model_Pooled_Summary.csv
cmgt_outputs/q1_validation/loco/Table_C5_LOCO_Paired_Statistics.csv
cmgt_outputs/q1_validation/loco/Table_C8_LOCO_Ablation_Pooled_Summary.csv
cmgt_outputs/q1_validation/loco/Table_C9_LOCO_Per_Year_Pooled_Performance.csv

Backbone:
cmgt_outputs/q1_validation/loco/backbone/Table_CB3_Backbone_Pooled_Summary.csv
cmgt_outputs/q1_validation/loco/backbone/Table_CB4_DINOv3_vs_Backbones_Paired_Statistics.csv

IMPORTANT INTERPRETATION
------------------------
LOCO contains unequal cohort sizes.

Therefore the script reports BOTH:

1. Macro fold mean:
   Each image-sharing cohort has equal weight.

2. Pooled out-of-fold performance:
   Every plant-year has equal weight because every plant-year is tested once.

For the manuscript's single overall predictive-performance number, the pooled
out-of-fold table is usually the more intuitive primary result, while the
macro fold table shows variation across image cohorts.

The fold audit must show:
shared_image_paths = 0
for every fold.
