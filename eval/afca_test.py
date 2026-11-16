"""OPTIONAL: how well does the US-trained model do on Australian AFCA decisions?
Needs afca/afca_100.csv with columns: text,label,link   (label 1 = complainant got money back)
Run:   python -m eval.afca_test
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from claimback.predict import predict_batch

df = pd.read_csv("afca/afca_100.csv").dropna(subset=["text", "label"])
p = np.array([r["p_win"] for r in predict_batch(df.text.tolist())])
y = df.label.astype(int).values
auc = roc_auc_score(y, p)
rng = np.random.default_rng(0)
boots = []
for _ in range(1000):                                   # bootstrap: resample to get a range
    idx = rng.integers(0, len(y), len(y))
    if len(set(y[idx])) == 2:
        boots.append(roc_auc_score(y[idx], p[idx]))
res = {"n": int(len(y)), "auc": auc, "auc_low_95": float(np.percentile(boots, 2.5)),
       "auc_high_95": float(np.percentile(boots, 97.5))}
json.dump(res, open("eval/afca_transfer.json", "w"), indent=1)
print(json.dumps(res, indent=1))
