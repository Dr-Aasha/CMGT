#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from pathlib import Path
os.environ.setdefault("MPLBACKEND","Agg")
import matplotlib; matplotlib.use("Agg",force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from q1_validation_common import *
from src.backbones import build_embedding_cache
from src.sensors import SensorRepository
from src.trajectory_features import build_cmgt_samples

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repeats",type=int,default=20)
    ap.add_argument("--no-safe-rerun",action="store_true",
                    help="Only audit overlap; do not rerun the complete model suite on image-cohort-safe splits.")
    args=ap.parse_args()

    cfg=load_config("config.yaml"); seed_everything(cfg["seed"])
    manifest=load_manifest(cfg)
    out=ensure_dir(Path(cfg["data"]["output_dir"])/"q1_validation"/"leakage")

    g2i,i2g,unresolved=build_group_image_map(manifest,cfg)
    counts=np.asarray([len(v) for v in i2g.values()],int)
    gy=manifest[["group","year"]].drop_duplicates().set_index("group")["year"].to_dict()

    shared=[]
    for p,gs in i2g.items():
        years=sorted({int(gy[g]) for g in gs})
        if len(gs)>1:
            shared.append({"image_path":p,"n_plant_year_groups":len(gs),"n_years":len(years),
                           "years":";".join(map(str,years)),"groups":";".join(sorted(gs)),
                           "cross_year_shared":len(years)>1})
    shared=pd.DataFrame(shared)
    summary=pd.DataFrame([{
        "plant_year_groups":manifest["group"].nunique(),
        "unique_resolved_images":len(i2g),
        "images_linked_to_multiple_groups":int(np.sum(counts>1)),
        "fraction_images_shared_across_groups":float(np.mean(counts>1)) if len(counts) else 0,
        "max_groups_per_image":int(np.max(counts)) if len(counts) else 0,
        "cross_year_shared_images":int(shared["cross_year_shared"].sum()) if len(shared) else 0,
        "groups_with_at_least_one_image":len(g2i),
        "groups_without_resolved_images":manifest["group"].nunique()-len(g2i)
    }])
    save_csv(summary,out/"Table_L1_Global_Image_Sharing_Summary.csv")
    save_csv(shared,out/"Table_L2_Shared_Images_Across_PlantYears.csv")
    save_csv(unresolved,out/"Unresolved_Image_References.csv")

    std=standard_splits(manifest,args.repeats,cfg["experiment"]["test_fraction"],cfg["seed"])
    std_a,std_d=audit_splits(std,g2i,"original_year_stratified")
    save_csv(std_a,out/"Table_L3_Original_Split_Image_Overlap.csv")
    save_csv(std_d,out/"Original_Split_Shared_Image_Paths.csv")

    # LOYO audit
    meta=manifest[["group","year"]].drop_duplicates("group")
    lrows=[]; ldetail=[]
    for year in sorted(meta["year"].unique()):
        tr=meta.loc[meta["year"]!=year,"group"].astype(str).tolist()
        te=meta.loc[meta["year"]==year,"group"].astype(str).tolist()
        a,d=audit_splits([{"repeat":int(year),"train_groups":tr,"test_groups":te}],g2i,f"LOYO_{year}")
        row=a.iloc[0].to_dict(); row["heldout_year"]=int(year); lrows.append(row)
        if len(d):
            d["heldout_year"]=int(year); ldetail.append(d)
    save_csv(pd.DataFrame(lrows),out/"Table_L4_LOYO_Image_Overlap.csv")
    save_csv(pd.concat(ldetail,ignore_index=True) if ldetail else pd.DataFrame(),
             out/"LOYO_Shared_Image_Paths.csv")

    cmap,cohorts=build_components(manifest,i2g)
    save_csv(cohorts,out/"Table_L5_Image_Sharing_Cohorts.csv")
    safe=safe_splits(manifest,cmap,args.repeats,cfg["experiment"]["test_fraction"],cfg["seed"])
    safe_a,safe_d=audit_splits(safe,g2i,"image_cohort_safe")
    save_csv(safe_a,out/"Table_L5b_LeakageSafe_Split_Image_Overlap.csv")
    save_csv(safe_d,out/"LeakageSafe_Shared_Image_Paths.csv")

    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.bar(std_a["repeat"].astype(int)+1,std_a["shared_image_paths"])
    ax.set_xlabel("Repeated split"); ax.set_ylabel("Shared raw image paths")
    ax.set_title("Train-test raw-image overlap in original split protocol")
    ax.grid(axis="y",alpha=.25); fig.tight_layout()
    fig.savefig(out/"Figure_L1_Original_Split_Image_Overlap.png",dpi=600,bbox_inches="tight"); plt.close(fig)

    maxov=int(std_a["shared_image_paths"].max())
    text=[
        "CMGT-DINOv3 IMAGE-OVERLAP / LEAKAGE AUDIT",
        "==========================================",
        f"Unique resolved images: {int(summary.iloc[0]['unique_resolved_images'])}",
        f"Images linked to >1 plant-year group: {int(summary.iloc[0]['images_linked_to_multiple_groups'])}",
        f"Maximum plant-year groups linked to one image: {int(summary.iloc[0]['max_groups_per_image'])}",
        f"Cross-year shared images: {int(summary.iloc[0]['cross_year_shared_images'])}",
        f"Maximum identical train/test image paths in original repeated splits: {maxov}",
        f"Maximum identical train/test image paths in safe splits: {int(safe_a['shared_image_paths'].max())}",
        "",
        ("RESULT: No raw-image path leakage was detected in the reconstructed original splits."
         if maxov==0 else
         "RESULT: Raw-image path overlap exists in the original splits. Use the image-cohort-safe re-evaluation in the paper.")
    ]
    (out/"Leakage_Audit_Interpretation.txt").write_text("\n".join(text),encoding="utf-8")

    if not args.no_safe_rerun:
        print("[SAFE RERUN] Loading cached DINOv3 / building features...")
        emb,emeta=build_embedding_cache(manifest,cfg,backbone_name="dinov3_vits16")
        srepo=SensorRepository(cfg["data"]["root"],cfg["data"]["sensor_features"])
        data=build_cmgt_samples(manifest,emb,emeta,srepo,cfg)
        met,summ=evaluate_suite(data,safe,cfg)
        sdir=ensure_dir(Path(cfg["data"]["output_dir"])/"q1_validation"/"leakage_safe_evaluation")
        save_csv(met,sdir/"Table_L6_LeakageSafe_Model_Metrics_All_Repeats.csv")
        save_csv(summ,sdir/"Table_L7_LeakageSafe_Model_Summary.csv")
        fig,ax=plt.subplots(figsize=(10,6.3))
        q=summ.sort_values("RMSE_mean")
        ax.barh(q["model"],q["RMSE_mean"],xerr=q["RMSE_std"],capsize=3)
        ax.set_xlabel("RMSE"); ax.set_title("Image-cohort-safe repeated evaluation"); ax.grid(axis="x",alpha=.25)
        fig.tight_layout(); fig.savefig(sdir/"Figure_L2_LeakageSafe_Model_RMSE.png",dpi=600,bbox_inches="tight"); plt.close(fig)

    print("\nDONE. Review:")
    print(out/"Table_L3_Original_Split_Image_Overlap.csv")
    print(out/"Table_L5b_LeakageSafe_Split_Image_Overlap.csv")
    if not args.no_safe_rerun:
        print(Path(cfg["data"]["output_dir"])/"q1_validation"/"leakage_safe_evaluation"/"Table_L7_LeakageSafe_Model_Summary.csv")

if __name__=="__main__": main()
