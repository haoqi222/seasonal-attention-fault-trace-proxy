#!/usr/bin/env python3
"""Validate and aggregate corrected V1.1 random-holdout inference."""

from __future__ import annotations

import csv, hashlib, json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROTOCOL=Path("03_configs/random_holdout_v1_1_inference_protocol.json")
OUTPUT=Path("04_results/random_holdout_v1_1_aggregate")

def sha256(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as h:
        for b in iter(lambda:h.read(8*1024*1024),b""): d.update(b)
    return d.hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write_json(path,value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def csv_rows(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def write_csv(path,values):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(values[0]),lineterminator="\n");w.writeheader();w.writerows(values)
def metrics(y,p,m):
    t=(y[m]==1).astype(np.int8);s=np.asarray(p[m],dtype=np.float64)
    out={"n":int(len(t)),"positive":int(t.sum()),"positive_rate":float(t.mean()),
         "average_precision":float(average_precision_score(t,s)),"roc_auc":float(roc_auc_score(t,s))}
    for f in (.01,.05):
        n=max(1,int(math.ceil(len(s)*f)));top=np.argpartition(s,-n)[-n:];tp=int(t[top].sum());label=f"top_{int(f*100)}pct"
        out[f"precision_{label}"]=tp/n;out[f"recall_{label}"]=tp/max(int(t.sum()),1)
    pred=s>=.5;tp=int((pred&(t==1)).sum());fp=int((pred&(t==0)).sum());fn=int((~pred&(t==1)).sum())
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    out.update({"threshold":.5,"f1_at_0_5":2*precision*recall/max(precision+recall,1e-9),
                "precision_at_0_5":precision,"recall_at_0_5":recall,"tp_at_0_5":tp,"fp_at_0_5":fp,"fn_at_0_5":fn})
    return out

def main():
    root=Path(__file__).resolve().parents[2];protocol_path=root/PROTOCOL;protocol=load(protocol_path);digest=sha256(protocol_path)
    parent_sha=protocol["parent_random_training_protocol_sha256"];parent_protocol=load(root/"03_configs/random_holdout_v1_protocol.json")
    split_records={x["split_id"]:x for x in parent_protocol["splits"]};raw=root/"01_data/interim/results_random_holdout_v1_1"/digest
    output=root/OUTPUT
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    data={}
    for split,record in split_records.items():
        y=np.load(root/"01_data/interim/dataset_v3"/record["source_fold"]/"y.npy",mmap_mode="r")
        folder=root/"01_data/interim/random_holdout_v1"/parent_sha/split
        data[split]={"y":y,"validation":np.load(folder/"val.npy",mmap_mode="r").astype(bool),
                     "test":np.load(folder/"test.npy",mmap_mode="r").astype(bool)}
    expected={"config_snapshot.json","environment.json","probability.npy","pr_curve.npz","result.json","run_manifest.json"}
    inventory=[];individual=[];maximum=0.0
    for model in protocol["models"]:
      for split in protocol["splits"]:
       for seed in protocol["seeds"]:
        cell=raw/model/split/f"seed_{seed}";errors=[]
        actual={p.name for p in cell.iterdir() if p.is_file()} if cell.is_dir() else set()
        if actual!=expected:errors.append("file_set")
        try:
            manifest=load(cell/"run_manifest.json");result=load(cell/"result.json")
            identity={"status":"complete","scope":"formal_random_holdout_v1_1_inference","protocol_sha256":digest,
                      "model_id":model,"split_id":split,"seed":seed}
            if any(manifest.get(k)!=v for k,v in identity.items()):errors.append("identity")
            if set(manifest.get("file_sha256",{}))!=expected-{"run_manifest.json"}:errors.append("hash_list")
            for name,value in manifest.get("file_sha256",{}).items():
                if sha256(cell/name)!=value:errors.append("hash:"+name)
            if load(cell/"config_snapshot.json")!=protocol:errors.append("snapshot")
            parent=root/"01_data/interim/results_random_holdout_v1"/parent_sha/model/split/f"seed_{seed}/checkpoint.pt"
            if sha256(parent)!=result["parent_checkpoint_sha256"]:errors.append("parent_checkpoint")
            p=np.load(cell/"probability.npy",mmap_mode="r")
            if p.shape!=data[split]["y"].shape or not np.isfinite(p).all() or p.min()<0 or p.max()>1:errors.append("probability")
            row={"model_id":model,"split_id":split,"source_fold":split_records[split]["source_fold"],"seed":seed}
            for role in ("validation","test"):
                calc=metrics(data[split]["y"],p,data[split][role])
                for k,v in calc.items():
                    if k in {"n","positive","tp_at_0_5","fp_at_0_5","fn_at_0_5"}:
                        if int(result[role][k])!=int(v):errors.append("metric:"+role+":"+k)
                    else:maximum=max(maximum,abs(float(result[role][k])-float(v)))
                    row[f"{role}_{k}"]=v
            individual.append(row)
        except Exception as e:errors.append(type(e).__name__+":"+str(e))
        inventory.append({"model_id":model,"split_id":split,"seed":seed,"status":"passed" if not errors else "failed","errors":" | ".join(errors)})
    write_csv(output/"run_inventory_random_v1_1.csv",inventory)
    failures=[x for x in inventory if x["status"]!="passed"]
    if failures:raise RuntimeError(f"{len(failures)} cells failed validation")
    if maximum>5e-12:raise RuntimeError(f"Metric delta {maximum}")
    write_csv(output/"individual_metrics_random_v1_1.csv",individual)
    ensembles=[]
    for model in protocol["models"]:
      for split in protocol["splits"]:
        total=np.zeros(data[split]["y"].shape,dtype=np.float64)
        for seed in protocol["seeds"]:total+=np.load(raw/model/split/f"seed_{seed}/probability.npy",mmap_mode="r")
        p=total/4;row={"model_id":model,"split_id":split,"source_fold":split_records[split]["source_fold"],"seed_count":4}
        for role in ("validation","test"):
            for k,v in metrics(data[split]["y"],p,data[split][role]).items():row[f"{role}_{k}"]=v
        ensembles.append(row)
    write_csv(output/"split_seed_ensemble_metrics_random_v1_1.csv",ensembles)
    summaries=[]
    for model in protocol["models"]:
        rs=[x for x in ensembles if x["model_id"]==model];aps=np.array([x["test_average_precision"] for x in rs]);aucs=np.array([x["test_roc_auc"] for x in rs])
        summaries.append({"model_id":model,"split_count":4,"seed_count_per_split":4,
                          "test_ap_mean":float(aps.mean()),"test_ap_sd":float(aps.std(ddof=1)),
                          "test_roc_auc_mean":float(aucs.mean()),"test_roc_auc_sd":float(aucs.std(ddof=1))})
    summaries.sort(key=lambda x:x["test_ap_mean"],reverse=True)
    for rank,row in enumerate(summaries,1):row["descriptive_rank"]=rank
    write_csv(output/"model_summary_random_v1_1.csv",summaries)
    spatial={(x["model_id"],x["fold_id"]):x for x in csv_rows(root/"04_results/experiment_v4_aggregate/fold_seed_ensemble_metrics_v4.csv")}
    old={(x["model_id"],x["split_id"]):x for x in csv_rows(root/"04_results/random_holdout_v1_aggregate/split_seed_ensemble_metrics_random_holdout_v1.csv")}
    paired=[];changes=[]
    for row in ensembles:
        model,split,fold=row["model_id"],row["split_id"],row["source_fold"];ap=float(row["test_average_precision"])
        spatial_ap=float(spatial[(model,fold)]["test_average_precision"]);old_ap=float(old[(model,split)]["test_average_precision"])
        paired.append({"model_id":model,"split_id":split,"source_fold":fold,"random_v1_1_test_ap":ap,
                       "spatial_v4_2_test_ap":spatial_ap,"random_minus_spatial_ap":ap-spatial_ap,
                       "random_to_spatial_ap_ratio":ap/spatial_ap})
        changes.append({"model_id":model,"split_id":split,"v1_floor_0_05_ap":old_ap,"v1_1_floor_0_01_ap":ap,
                        "v1_1_minus_v1_ap":ap-old_ap})
    write_csv(output/"spatial_random_paired_comparison_v1_1.csv",paired);write_csv(output/"v1_v1_1_ap_changes.csv",changes)
    acceptance={"status":"accepted","protocol_sha256":digest,"parent_training_protocol_sha256":parent_sha,
                "validated_cells":48,"failed_cells":0,"seed_ensembles":12,"maximum_metric_recalculation_delta":maximum,
                "blend_floor":protocol["blend"]["floor"],"matched_spatial_protocol_sha256":protocol["blend"]["matched_spatial_protocol_sha256"],
                "supersedes":"V1 floor-0.05 probability maps for final comparison; training checkpoints unchanged",
                "next_gate":"independent checksum and ensemble recomputation"}
    write_json(output/"acceptance_random_v1_1.json",acceptance)
    (output/"README.md").write_text("# Random-holdout diagnostic V1.1\n\nEvaluation-only correction using the exact V4.2 blend floor of 0.01.\n",encoding="utf-8")
    names=sorted(p.name for p in output.iterdir() if p.is_file() and p.name!="checksums_random_v1_1.sha256")
    (output/"checksums_random_v1_1.sha256").write_text("".join(f"{sha256(output/n)}  {n}\n" for n in names),encoding="ascii")
    print(json.dumps(acceptance,indent=2))
if __name__=="__main__":main()
