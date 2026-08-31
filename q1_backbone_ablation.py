#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from pathlib import Path
os.environ.setdefault("MPLBACKEND","Agg")
import matplotlib; matplotlib.use("Agg",force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from q1_validation_common import *
from src.backbones import build_embedding_cache
from src.sensors import SensorRepository
from src.trajectory_features import build_cmgt_samples

DEFAULT=["efficientnet_b0","convnext_tiny","dinov2_small","dinov3_vits16"]

def evaluate_backbone(data,splits,cfg,bkey):
    rows=[]; disp=BACKBONE_DISPLAY.get(bkey,bkey)
    for s in splits:
        rep=int(s["repeat"]); seed=int(cfg["seed"])+300000+rep
        tr=groups_to_indices(data,s["train_groups"]); te=groups_to_indices(data,s["test_groups"])
        pp=CMGTPreprocessor(cfg).fit(data,tr); ytr=data["y"][tr]; yte=data["y"][te]

        # Static image
        a=pp.transform_block(data,tr,"static_image"); b=pp.transform_block(data,te,"static_image")
        p=fit_extratrees(a,ytr,seed).predict(b)
        rows.append({"backbone":disp,"backbone_key":bkey,"protocol":"Static Image ExtraTrees","repeat":rep,
                     **regression_metrics(yte,p)})

        # Image trajectory only
        a=pp.transform_block(data,tr,"image_trajectory"); b=pp.transform_block(data,te,"image_trajectory")
        p=fit_extratrees(a,ytr,seed+1).predict(b)
        rows.append({"backbone":disp,"backbone_key":bkey,"protocol":"Image Trajectory ExtraTrees","repeat":rep,
                     **regression_metrics(yte,p)})

        # Full CMGT
        m=fit_full_cmgt(data,pp,tr,cfg,seed+2)
        p=m.predict(pp.proposed_matrix(data,te))
        rows.append({"backbone":disp,"backbone_key":bkey,"protocol":"Full CMGT","repeat":rep,
                     **regression_metrics(yte,p)})
        print(f"[{disp}] {rep+1}/{len(splits)} Full-CMGT RMSE={rows[-1]['RMSE']:.4f}")
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repeats",type=int,default=20)
    ap.add_argument("--split-mode",choices=["safe","standard"],default="safe")
    ap.add_argument("--backbones",nargs="+",default=DEFAULT)
    args=ap.parse_args()

    cfg=load_config("config.yaml"); seed_everything(cfg["seed"])
    manifest=load_manifest(cfg)
    out=ensure_dir(Path(cfg["data"]["output_dir"])/"q1_validation"/"backbone_ablation")

    g2i,i2g,_=build_group_image_map(manifest,cfg)
    cmap,cohorts=build_components(manifest,i2g)
    if args.split_mode=="safe":
        splits=safe_splits(manifest,cmap,args.repeats,cfg["experiment"]["test_fraction"],cfg["seed"])
        audit,_=audit_splits(splits,g2i,"image_cohort_safe")
        if int(audit["shared_image_paths"].max())!=0:
            raise RuntimeError("Safe split construction failed: train/test image overlap is not zero.")
    else:
        splits=standard_splits(manifest,args.repeats,cfg["experiment"]["test_fraction"],cfg["seed"])

    # Save exact group assignment used by every backbone
    srows=[]
    for s in splits:
        for role,gs in [("train",s["train_groups"]),("test",s["test_groups"])]:
            srows += [{"repeat":s["repeat"],"role":role,"group":g} for g in gs]
    save_csv(pd.DataFrame(srows),out/"Table_B0_Backbone_Ablation_Split_Assignments.csv")

    srepo=SensorRepository(cfg["data"]["root"],cfg["data"]["sensor_features"])
    allres=[]
    for b in args.backbones:
        if b not in cfg["vision"]["backbones"]:
            raise KeyError(f"{b} is not defined under vision.backbones in config.yaml")
        print("\n"+"="*72); print("BACKBONE:",BACKBONE_DISPLAY.get(b,b)); print("="*72)
        emb,emeta=build_embedding_cache(manifest,cfg,backbone_name=b)
        data=build_cmgt_samples(manifest,emb,emeta,srepo,cfg)
        r=evaluate_backbone(data,splits,cfg,b)
        r["embedding_dim"]=int(emb.shape[1]); r["split_mode"]=args.split_mode
        allres.append(r)

    df=pd.concat(allres,ignore_index=True)
    summ=df.groupby(["protocol","backbone","backbone_key","embedding_dim"],as_index=False).agg(
        RMSE_mean=("RMSE","mean"),RMSE_std=("RMSE","std"),MAE_mean=("MAE","mean"),
        MAE_std=("MAE","std"),R2_mean=("R2","mean"),NRMSE_mean=("NRMSE","mean"),
        sMAPE_mean=("sMAPE","mean")).sort_values(["protocol","RMSE_mean"])
    save_csv(df,out/"Table_B1_Backbone_Ablation_All_Repeats.csv")
    save_csv(summ,out/"Table_B2_Backbone_Ablation_Summary.csv")

    # Full-CMGT statistics: DINOv3 vs each alternative
    q=df[df["protocol"]=="Full CMGT"]
    piv=q.pivot(index="repeat",columns="backbone",values="RMSE").dropna()
    fstat,fp=friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
    save_csv(pd.DataFrame([{"friedman_chi2":fstat,"p_value":fp,
                            "n_backbones":piv.shape[1],"n_repeats":piv.shape[0]}]),
             out/"Table_B3_Backbone_Friedman.csv")
    ref="DINOv3 ViT-S/16"; rows=[]; raw=[]
    for other in piv.columns:
        if other==ref: continue
        d=piv[ref].to_numpy()-piv[other].to_numpy()
        try:W,p=wilcoxon(d,zero_method="wilcox")
        except Exception:W,p=np.nan,1.0
        lo,hi=bootstrap_ci(d,5000,cfg["seed"])
        rows.append({"reference":ref,"comparison":other,
                     "mean_RMSE_difference_reference_minus_other":float(np.mean(d)),
                     "wilcoxon_W":W,"p_value":p,"cohen_dz":cohen_dz(d),
                     "bootstrap95_low":lo,"bootstrap95_high":hi})
        raw.append(p)
    adj=holm_adjust(raw)
    for r,h in zip(rows,adj):
        r["holm_p"]=float(h); r["significant_0.05"]=bool(h<.05)
    save_csv(pd.DataFrame(rows),out/"Table_B4_DINOv3_vs_Backbones_Paired_Statistics.csv")

    # Figure 1: Full CMGT
    full=summ[summ["protocol"]=="Full CMGT"].sort_values("RMSE_mean")
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    ax.barh(full["backbone"],full["RMSE_mean"],xerr=full["RMSE_std"],capsize=3)
    ax.set_xlabel("RMSE"); ax.set_title(f"Backbone ablation: full CMGT ({args.split_mode} splits)")
    ax.grid(axis="x",alpha=.25); fig.tight_layout()
    fig.savefig(out/"Figure_B1_Full_CMGT_Backbone_RMSE.png",dpi=600,bbox_inches="tight"); plt.close(fig)

    # Figure 2: three protocols
    order=[BACKBONE_DISPLAY.get(b,b) for b in args.backbones]
    protocols=["Static Image ExtraTrees","Image Trajectory ExtraTrees","Full CMGT"]
    fig,ax=plt.subplots(figsize=(10,6))
    x=np.arange(len(order)); width=.24
    for j,p in enumerate(protocols):
        z=summ[summ["protocol"]==p].set_index("backbone").reindex(order)
        ax.bar(x+(j-1)*width,z["RMSE_mean"],width=width,yerr=z["RMSE_std"],capsize=2,label=p)
    ax.set_xticks(x,order,rotation=15); ax.set_ylabel("RMSE")
    ax.set_title(f"Visual backbone comparison under identical {args.split_mode} splits")
    ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout()
    fig.savefig(out/"Figure_B2_Backbone_Protocols_RMSE.png",dpi=600,bbox_inches="tight"); plt.close(fig)

    print("\nDONE. Most important files:")
    print(out/"Table_B2_Backbone_Ablation_Summary.csv")
    print(out/"Table_B4_DINOv3_vs_Backbones_Paired_Statistics.csv")

if __name__=="__main__": main()
