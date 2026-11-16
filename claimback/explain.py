"""Which sentences help or hurt the case?

Method: remove one sentence at a time and re-score. If the score drops without
a sentence, that sentence was helping ("+"). If it rises, it was hurting ("-").
"""
import re

from claimback.predict import raw_win_scores


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p for p in parts if p.strip()]


def sentence_impact(text, threshold=0.02):
    """Returns a list like [("I reported it the same day. ", "+"), ("It's unfair. ", None)]
    which is exactly what Gradio's HighlightedText box wants."""
    sents = split_sentences(text)
    if len(sents) < 2:
        return [(text, None)]
    variants = [text] + [" ".join(sents[:i] + sents[i + 1:]) for i in range(len(sents))]
    scores = raw_win_scores(variants)
    base = scores[0]
    out = []
    for s, without in zip(sents, scores[1:]):
        diff = base - without
        label = "+" if diff > threshold else "-" if diff < -threshold else None
        out.append((s + " ", label))
    return out


if __name__ == "__main__":                          # quick test:  python -m claimback.explain
    print(sentence_impact("I lost $500. I reported it the same day. This is so unfair."))
