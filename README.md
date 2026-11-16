---
title: Claimback
emoji: 💸
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Claimback
# [Website] https://claimback-9e2.pages.dev/ 
Claimback helps people wronged by their bank get their money back by predicting whether their
complaint will win money, and coaching them to rewrite it so it looks like the complaints that did.

**Follow the step-by-step guide PDF (claimback-complete-guide.pdf). This file is the short version.**

## Run the app on your computer
```
python -m venv .venv
# Windows:  .venv\Scripts\activate        Mac:  source .venv/bin/activate
pip install -r requirements.txt
python app.py            # then open http://127.0.0.1:7860
```
Copy `.env.example` to `.env` and paste your Gemini API key to turn on the AI checklist and rewrite.

## What's where
| Path | What it is | Owner |
|---|---|---|
| `app.py` | the web app | B |
| `claimback/config.py` | ALL settings (which model, model names, LLM) | both |
| `claimback/predict.py` | scores a complaint (stub / tfidf / fine-tuned) | A |
| `claimback/explain.py` | which sentences help or hurt | A |
| `claimback/checklist.py` | evidence checklist + gap questions | B |
| `claimback/rewrite.py` | rewrite + fact guards | B |
| `claimback/letters.py` | letters + base rates | B |
| `notebooks/claimback_train.ipynb` | the ONE Kaggle notebook: data, baseline, fine-tuning, evaluation | A |
| `eval/` | uplift test and optional AFCA test | A |
| `demo_stories/` | 6 fictional stories for the demo | B |

## Data and credits
- CFPB Consumer Complaint Database (US Consumer Financial Protection Bureau), public data.
- Optional: AFCA published determinations, credited to the Australian Financial Complaints Authority, with links.
- Not legal advice.
