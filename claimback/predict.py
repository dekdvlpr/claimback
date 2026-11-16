"""Scores a complaint: chance of getting money back + issue type.

THE CONTRACT (never change these two function names or what they return):
    predict(text)        -> {"p_win": 0.47, "issue": "...", "issue_probs": {...}, "model_version": "..."}
    predict_batch(texts) -> a list of the same dicts

Switch between the fake model, the baseline and the real model in config.py (MODEL_BACKEND).
"""
import functools
import os
import re

from claimback import config


# ----------------------------- 1. stub (fake) -------------------------------
_HELPFUL_WORDS = [
    ("reported", .08), ("same day", .08), ("immediately", .06), ("$", .05),
    ("did not authori", .10), ("didn't authori", .10), ("never authori", .10),
    ("no warning", .07), ("reference", .05), ("case number", .05), ("refund", .04),
    ("on ", .02), ("statement", .03), ("dispute", .04),
]


def _stub_one(text):
    t = text.lower()
    score = 0.12 + sum(w for kw, w in _HELPFUL_WORDS if kw in t)
    score += min(len(t) / 4000, 0.10)              # longer, detailed stories score a bit higher
    if re.search(r"scam|fraud|authori|hack|stole|text from|link|log ?in|money was gone", t):
        issue = "Scam or unauthorised payment"
    elif re.search(r"fee|overdraft|charge", t):
        issue = "Fees and charges"
    elif re.search(r"collector|collection", t):
        issue = "Debt collection"
    elif re.search(r"mortgage|loan", t):
        issue = "Loan or mortgage servicing"
    elif re.search(r"dispute|merchant|purchase", t):
        issue = "Card or transaction dispute"
    else:
        issue = "Account problems"
    probs = {g: (0.7 if g == issue else 0.3 / (len(config.ISSUE_GROUPS) - 1)) for g in config.ISSUE_GROUPS}
    return {"p_win": round(min(score, 0.9), 3), "issue": issue, "issue_probs": probs, "model_version": "stub-0"}


# ----------------------------- 2. TF-IDF baseline ---------------------------
@functools.lru_cache(maxsize=1)
def _tfidf():
    import joblib
    return joblib.load(config.TFIDF_PATH)          # dict made by the Kaggle notebook (Step 10)


def _tfidf_batch(texts):
    m = _tfidf()
    X = m["vectorizer"].transform(texts)
    p_win = m["win_model"].predict_proba(X)[:, 1]
    issue_p = m["issue_model"].predict_proba(X)
    out = []
    for pw, ip in zip(p_win, issue_p):
        probs = dict(zip(m["issue_model"].classes_, map(float, ip)))
        out.append({"p_win": float(pw), "issue": max(probs, key=probs.get),
                    "issue_probs": probs, "model_version": "tfidf-0"})
    return out


# ----------------------------- 3. fine-tuned (Hugging Face) -----------------
@functools.lru_cache(maxsize=2)
def _hf(name):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    return AutoTokenizer.from_pretrained(name), AutoModelForSequenceClassification.from_pretrained(name).eval()


def _hf_probs(name, texts):
    import torch
    tok, model = _hf(name)
    with torch.no_grad():
        x = tok(texts, truncation=True, max_length=384, padding=True, return_tensors="pt")
        return torch.softmax(model(**x).logits, -1).tolist()


def _hf_batch(texts):
    win = _hf_probs(config.HF_WIN_MODEL, texts)
    iss = _hf_probs(config.HF_ISSUE_MODEL, texts)
    out = []
    for w, i in zip(win, iss):
        probs = dict(zip(config.ISSUE_GROUPS, i))
        out.append({"p_win": float(w[1]), "issue": max(probs, key=probs.get),
                    "issue_probs": probs, "model_version": "finetuned"})
    return out


# ----------------------------- calibration ----------------------------------
@functools.lru_cache(maxsize=1)
def _calibrator():
    if not os.path.exists(config.CALIBRATOR_PATH):
        return None
    import joblib
    return joblib.load(config.CALIBRATOR_PATH)


def _calibrate(results):
    # The calibrator is fitted on the FINE-TUNED model's scores, so it is only used with "hf".
    cal = _calibrator()
    if cal is None or config.MODEL_BACKEND != "hf":
        return results
    raw = [r["p_win"] for r in results]
    for r, p in zip(results, cal.predict(raw)):
        r["p_win"] = round(float(p), 3)
    return results


def _clip(results):
    # Never show 0% or 100%: no model is ever that sure about a real person's case.
    for r in results:
        r["p_win"] = round(min(max(float(r["p_win"]), 0.02), 0.98), 3)
    return results


# ----------------------------- public functions -----------------------------
def predict_batch(texts):
    texts = [t if t and t.strip() else "(empty)" for t in texts]
    if config.MODEL_BACKEND == "hf":
        out = []
        for i in range(0, len(texts), 16):          # 16 at a time keeps memory low on a laptop
            out += _hf_batch(texts[i:i + 16])
        return _clip(_calibrate(out))
    if config.MODEL_BACKEND == "tfidf":
        return _clip(_tfidf_batch(texts))
    return [_stub_one(t) for t in texts]


def predict(text):
    return predict_batch([text])[0]


def raw_win_scores(texts):
    """Uncalibrated money-back scores only (faster: skips the issue model).
    Used by the sentence highlighter, which needs small, smooth differences."""
    texts = [t if t and t.strip() else "(empty)" for t in texts]
    if config.MODEL_BACKEND == "hf":
        out = []
        for i in range(0, len(texts), 16):
            out += [w[1] for w in _hf_probs(config.HF_WIN_MODEL, texts[i:i + 16])]
        return out
    if config.MODEL_BACKEND == "tfidf":
        m = _tfidf()
        return list(m["win_model"].predict_proba(m["vectorizer"].transform(texts))[:, 1])
    return [_stub_one(t)["p_win"] for t in texts]


if __name__ == "__main__":                          # quick test:  python -m claimback.predict
    print(predict("Someone took $900 from my account. I reported it the same day and never authorised it."))
