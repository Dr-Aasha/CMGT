from __future__ import annotations
import os, json
from collections import defaultdict
from pathlib import Path
os.environ.setdefault("MPLBACKEND","Agg")
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from src.utils import load_config, seed_everything, regression_metrics
from src.image_paths import split_photo_raw, resolve_image, build_basename_index
from src.evaluate import year_stratified_split
from src.modeling import CMGTPreprocessor, OOFBlendRegressor, fit_extratrees, fit_named

BACKBONE_DISPLAY = {
    "efficientnet_b0":"EfficientNet-B0",
    "convnext_tiny":"ConvNeXt-Tiny",
    "dinov2_small":"DINOv2-Small",
    "dinov3_vits16":"DINOv3 ViT-S/16",
}

def ensure_dir(p):
    p=Path(p); p.mkdir(parents=True, exist_ok=True); return p

def save_csv(df,p):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); df.to_csv(p,index=False); print("[SAVED]",p)

def load_manifest(cfg):
    p=Path(cfg["data"]["manifest_csv"])
    if not p.exists():
        for q in cfg["data"].get("external_manifest_candidates",[]):
            q=Path(q)
            if q.exists(): p=q; break
    if not p.exists():
        raise FileNotFoundError("Three-year manifest not found. Run: python3 prepare_manifest.py")
    df=pd.read_csv(p)
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    df["target"]=pd.to_numeric(df["target"],errors="coerce")
    df=df.dropna(subset=["target"]).copy()
    if "group" not in df.columns:
        df["group"]=df["year"].astype(int).astype(str)+"_"+df["plant_id"].astype(str)
    df["group"]=df["group"].astype(str)
    df["year"]=pd.to_numeric(df["year"],errors="coerce").astype(int)
    print("[MANIFEST]",len(df),"rows;",df["group"].nunique(),"plant-year groups;",
          sorted(df["year"].unique().tolist()))
    return df

def build_group_image_map(manifest,cfg):
    root=cfg["data"]["root"]
    bidx=build_basename_index(root)
    g2i=defaultdict(set); i2g=defaultdict(set); unresolved=[]
    declared=resolved=0
    for r in manifest.itertuples():
        g=str(r.group)
        raws=split_photo_raw(getattr(r,"photo_raw",None)); declared += len(raws)
        for raw in raws:
            p=resolve_image(raw,root,getattr(r,"manual_csv",None),bidx)
            if p:
                p=str(Path(p).resolve()); g2i[g].add(p); i2g[p].add(g); resolved+=1
            else:
                unresolved.append({"group":g,"year":int(r.year),"photo_raw":str(raw)})
    print("[IMAGE MAP] declared refs=",declared,"resolved refs=",resolved,
          "unique images=",len(i2g),"groups with images=",len(g2i))
    return dict(g2i),dict(i2g),pd.DataFrame(unresolved)

def image_union(groups,g2i):
    s=set()
    for g in groups: s.update(g2i.get(str(g),set()))
    return s

class UnionFind:
    def __init__(self,items):
        self.p={x:x for x in items}; self.rank={x:0 for x in items}
    def find(self,x):
        if self.p[x]!=x: self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.rank[a]<self.rank[b]: a,b=b,a
        self.p[b]=a
        if self.rank[a]==self.rank[b]: self.rank[a]+=1

def build_components(manifest,i2g):
    groups=sorted(manifest["group"].unique().tolist()); uf=UnionFind(groups)
    for gs in i2g.values():
        gs=sorted(set(gs))
        if len(gs)>1:
            for g in gs[1:]: uf.union(gs[0],g)
    comp=defaultdict(list)
    for g in groups: comp[uf.find(g)].append(g)
    gy=manifest[["group","year"]].drop_duplicates().set_index("group")["year"].to_dict()
    items=[]
    for members in comp.values():
        members=sorted(members); years=sorted({int(gy[g]) for g in members})
        items.append((years,members))
    items.sort(key=lambda z:(z[0],z[1][0]))
    cmap={}; rows=[]
    for k,(years,members) in enumerate(items,1):
        cid=f"C{k:04d}"
        for g in members:cmap[g]=cid
        rows.append({"component_id":cid,"n_groups":len(members),"n_years":len(years),
                     "years":";".join(map(str,years)),"groups":";".join(members),
                     "cross_year_component":len(years)>1})
    return cmap,pd.DataFrame(rows)

def choose_components(items,target,rng):
    items=list(items); rng.shuffle(items)
    # random priority with a mild preference for smaller cohorts
    items=sorted(items,key=lambda z:(len(z[1]),rng.random()))
    chosen=[]; n=0
    for cid,gs in items:
        if not chosen:
            chosen.append((cid,gs)); n+=len(gs)
            if n>=target: break
            continue
        old=abs(n-target); new=abs(n+len(gs)-target)
        if n<target and (new<=old or rng.random()<0.30):
            chosen.append((cid,gs)); n+=len(gs)
        if n>=target and abs(n-target)<=1: break
    have={c for c,_ in chosen}
    if n<target:
        rem=[x for x in items if x[0] not in have]; rng.shuffle(rem)
        for cid,gs in rem:
            chosen.append((cid,gs)); n+=len(gs)
            if n>=target: break
    if len(chosen)==len(items) and len(items)>1: chosen=chosen[:-1]
    return {c for c,_ in chosen}

def safe_splits(manifest,cmap,repeats,test_fraction,seed):
    meta=manifest[["group","year"]].drop_duplicates("group").sort_values("group").reset_index(drop=True)
    meta["component_id"]=meta["group"].map(cmap)
    if (meta.groupby("component_id")["year"].nunique()>1).any():
        raise RuntimeError("An image-sharing component spans multiple years. Inspect cohort audit before safe splitting.")
    by_year={}
    for year,gy in meta.groupby("year"):
        by_year[int(year)]=[(cid,c["group"].astype(str).tolist()) for cid,c in gy.groupby("component_id")]
    out=[]
    for rep in range(int(repeats)):
        rng=np.random.default_rng(seed+100000+rep); test=set()
        for year,items in sorted(by_year.items()):
            n=sum(len(gs) for _,gs in items); target=max(1,int(round(n*test_fraction)))
            chosen=choose_components(items,target,rng)
            for cid,gs in items:
                if cid in chosen:test.update(gs)
        allg=set(meta["group"]); out.append({"repeat":rep,"train_groups":sorted(allg-test),"test_groups":sorted(test)})
    return out

def standard_splits(manifest,repeats,test_fraction,seed):
    meta=manifest[["group","year"]].drop_duplicates("group").sort_values("group").reset_index(drop=True)
    out=[]
    for rep in range(int(repeats)):
        tr,te=year_stratified_split(meta,test_fraction,seed+rep)
        out.append({"repeat":rep,"train_groups":meta.iloc[tr]["group"].astype(str).tolist(),
                    "test_groups":meta.iloc[te]["group"].astype(str).tolist()})
    return out

def audit_splits(splits,g2i,label):
    rows=[]; details=[]
    for s in splits:
        tr=image_union(s["train_groups"],g2i); te=image_union(s["test_groups"],g2i); ov=tr&te
        rows.append({"split_name":label,"repeat":s["repeat"],"train_groups":len(s["train_groups"]),
                     "test_groups":len(s["test_groups"]),"train_unique_images":len(tr),
                     "test_unique_images":len(te),"shared_image_paths":len(ov),
                     "test_image_overlap_fraction":len(ov)/max(len(te),1),"zero_overlap":len(ov)==0})
        details += [{"repeat":s["repeat"],"image_path":p} for p in sorted(ov)]
    return pd.DataFrame(rows),pd.DataFrame(details)

def groups_to_indices(data,groups):
    gs=data["meta"]["group"].astype(str).to_numpy()
    return np.where(np.isin(gs,list(set(map(str,groups)))))[0]

def fit_full_cmgt(data,pp,tr,cfg,seed):
    X=pp.proposed_matrix(data,tr)
    return OOFBlendRegressor(cfg["model"]["candidate_heads"],folds=cfg["model"]["inner_folds"],
                             seed=seed,use_blend=cfg["model"].get("use_oof_blend",True)).fit(X,data["y"][tr])

def evaluate_suite(data,splits,cfg):
    rows=[]
    for s in splits:
        rep=int(s["repeat"]); seed=int(cfg["seed"])+200000+rep
        tr=groups_to_indices(data,s["train_groups"]); te=groups_to_indices(data,s["test_groups"])
        pp=CMGTPreprocessor(cfg).fit(data,tr); ytr=data["y"][tr]; yte=data["y"][te]
        P={}
        Xi0=pp.transform_block(data,tr,"static_image"); Xi1=pp.transform_block(data,te,"static_image")
        P["DINOv3 Static Image ExtraTrees"]=fit_extratrees(Xi0,ytr,seed).predict(Xi1)
        Xp0=pp.transform_block(data,tr,"static_phenotype"); Xp1=pp.transform_block(data,te,"static_phenotype")
        P["Phenotype-only ExtraTrees"]=fit_extratrees(Xp0,ytr,seed+1).predict(Xp1)
        Xe0=pp.transform_block(data,tr,"static_environment"); Xe1=pp.transform_block(data,te,"static_environment")
        P["Environment-only ExtraTrees"]=fit_extratrees(Xe0,ytr,seed+2).predict(Xe1)
        Xs0=pp.static_matrix(data,tr); Xs1=pp.static_matrix(data,te)
        P["Static Early Fusion ExtraTrees"]=fit_extratrees(Xs0,ytr,seed+3).predict(Xs1)
        for name,disp in [("xgboost","Static Early Fusion XGBoost"),
                          ("catboost","Static Early Fusion CatBoost"),
                          ("histgb","Static Early Fusion HistGB")]:
            try:P[disp]=fit_named(name,Xs0,ytr,seed+10).predict(Xs1)
            except Exception as e:print("[WARN]",disp,e)
        blocks=["image_trajectory","phenotype_trajectory","environment_trajectory","meta_reliability"]
        Xt0=pp.proposed_matrix(data,tr,include=blocks); Xt1=pp.proposed_matrix(data,te,include=blocks)
        P["Trajectory Fusion ExtraTrees"]=fit_extratrees(Xt0,ytr,seed+20).predict(Xt1)
        m=fit_full_cmgt(data,pp,tr,cfg,seed+30)
        P["CMGT-DINOv3"]=m.predict(pp.proposed_matrix(data,te))
        for name,p in P.items():
            rows.append({"repeat":rep,"model":name,"train_n":len(tr),"test_n":len(te),**regression_metrics(yte,p)})
        print(f"[SAFE EVAL] {rep+1}/{len(splits)} CMGT RMSE={regression_metrics(yte,P['CMGT-DINOv3'])['RMSE']:.4f}")
    df=pd.DataFrame(rows)
    summary=df.groupby("model").agg(RMSE_mean=("RMSE","mean"),RMSE_std=("RMSE","std"),
        MAE_mean=("MAE","mean"),R2_mean=("R2","mean"),NRMSE_mean=("NRMSE","mean"),
        sMAPE_mean=("sMAPE","mean")).reset_index().sort_values("RMSE_mean")
    return df,summary

def holm_adjust(pvals):
    pvals=np.asarray(pvals,float); m=len(pvals); order=np.argsort(pvals); out=np.zeros(m); run=0
    for rank,idx in enumerate(order):
        run=max(run,(m-rank)*pvals[idx]); out[idx]=min(1.0,run)
    return out

def cohen_dz(d):
    d=np.asarray(d,float); sd=np.std(d,ddof=1)
    return float(np.mean(d)/sd) if len(d)>1 and sd>1e-12 else np.nan

def bootstrap_ci(d,n=5000,seed=42):
    d=np.asarray(d,float); rng=np.random.default_rng(seed)
    vals=[np.mean(rng.choice(d,size=len(d),replace=True)) for _ in range(int(n))]
    return float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))
