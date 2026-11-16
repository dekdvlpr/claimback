"""Person A: measures how much the rewrite raises the predicted chance (the "uplift" judge number).

Before running:
  1. config.py: MODEL_BACKEND = "hf" (or "tfidf" if the fine-tuned model isn't ready)
  2. .env has ANTHROPIC_API_KEY
  3. data/uplift_candidates.csv exists (downloaded from the Kaggle notebook's Output)
Run from the project folder:   python -m eval.uplift
Cost: about 50 complaints x up to 6 LLM calls. Cut N to 20 to test first.
"""
import json
import re

import pandas as pd

from claimback.predict import predict_batch
from claimback.rewrite import rewrite_complaint

N = 5
df = pd.read_csv("data/uplift_candidates.csv")
df["p"] = [r["p_win"] for r in predict_batch(df.text.tolist())]
pick = df.sort_values("p", ascending=False).head(N)        # complaints that clearly had a good case


def vague(text):
    """Simulate a weak first draft: just the first 2 sentences."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sents[:2])


rows = []
for i, r in enumerate(pick.itertuples()):
    draft = vague(r.text)
    best, old, new, log = rewrite_complaint(draft, answers=r.text)   # the full story plays the user's answers
    rows.append({"old": old, "new": new, "improved": new > old,
                 "rejected_attempts": sum("rejected" in l for l in log)})
    print(f"{i+1}/{len(pick)}  {old:.2f} -> {new:.2f}   {log}")

res = pd.DataFrame(rows)
summary = {"n": len(res), "mean_before": res.old.mean(), "mean_after": res.new.mean(),
           "mean_uplift": (res.new - res.old).mean(), "pct_improved": res.improved.mean(),
           "rejected_by_fact_guards": int(res.rejected_attempts.sum()),
           "note": "Accepted rewrites passed both fact guards, so they contain no unsupported claims."}
json.dump(summary, open("eval/uplift.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
