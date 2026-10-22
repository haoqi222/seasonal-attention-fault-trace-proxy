#!/usr/bin/env python3
"""Independent checksum and ensemble verification for random V1.1."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def main():
    root=Path(__file__).resolve().parents[2];pp=root/"03_configs/random_holdout_v1_1_inference_protocol.json";p=json.loads(pp.read_text());d=sha(pp)
    out=root/"04_results/random_holdout_v1_1_aggregate";raw=root/"01_data/interim/results_random_holdout_v1_1"/d
    parent=p["parent_random_training_protocol_sha256"];base=root/"01_data/interim/random_holdout_v1"/parent
    parent_p=json.loads((root/"03_configs/random_holdout_v1_protocol.json").read_text());sr={x["split_id"]:x for x in parent_p["splits"]}
    checks=[]
    def check(name,value):checks.append({"name":name,"passed":bool(value)})
    lines=(out/"checksums_random_v1_1.sha256").read_text().splitlines();check("checksum_count",len(lines)==8)
    for line in lines:
        digest,name=line.split("  ",1);check("hash:"+name,sha(out/name)==digest)
    table={(x["model_id"],x["split_id"]):x for x in rows(out/"split_seed_ensemble_metrics_random_v1_1.csv")}
    summary={x["model_id"]:x for x in rows(out/"model_summary_random_v1_1.csv")}
    for model in p["models"]:
        aps=[];aucs=[]
        for split in p["splits"]:
            y=np.load(root/"01_data/interim/dataset_v3"/sr[split]["source_fold"]/"y.npy",mmap_mode="r")
            test=np.load(base/split/"test.npy",mmap_mode="r").astype(bool);total=np.zeros(y.shape,dtype=np.float64)
            for seed in p["seeds"]:
                cell=raw/model/split/f"seed_{seed}";m=json.loads((cell/"run_manifest.json").read_text())
                check(f"identity:{model}:{split}:{seed}",m["status"]=="complete" and m["protocol_sha256"]==d)
                for name,digest in m["file_sha256"].items():check(f"run_hash:{model}:{split}:{seed}:{name}",sha(cell/name)==digest)
                total+=np.load(cell/"probability.npy",mmap_mode="r")
            score=(total/4)[test];truth=(y[test]==1).astype(np.int8);ap=float(average_precision_score(truth,score));auc=float(roc_auc_score(truth,score))
            aps.append(ap);aucs.append(auc);stored=table[(model,split)]
            check(f"ap:{model}:{split}",abs(ap-float(stored["test_average_precision"]))<=1e-12)
            check(f"auc:{model}:{split}",abs(auc-float(stored["test_roc_auc"]))<=1e-12)
        check("summary_ap:"+model,abs(np.mean(aps)-float(summary[model]["test_ap_mean"]))<=1e-12)
        check("summary_auc:"+model,abs(np.mean(aucs)-float(summary[model]["test_roc_auc_mean"]))<=1e-12)
    failures=[x for x in checks if not x["passed"]];report={"status":"accepted" if not failures else "rejected","protocol_sha256":d,
      "checks_passed":len(checks)-len(failures),"checks_total":len(checks),"failures":failures}
    (out/"independent_verification_random_v1_1.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2));raise SystemExit(1 if failures else 0)
if __name__=="__main__":main()
