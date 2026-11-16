"""Claimback web app.  Run it with:   python app.py   then open http://127.0.0.1:7860"""
import gradio as gr

from claimback import config
from claimback.checklist import all_labels, check_and_questions
from claimback.explain import sentence_impact
from claimback.letters import bank_letter, base_rate_text, regulator_version
from claimback.predict import predict
from claimback.rewrite import rewrite_complaint

DEMO_STORIES = {}
try:
    import pathlib
    for f in sorted(pathlib.Path("demo_stories").glob("*.txt")):
        DEMO_STORIES[f.stem.replace("_", " ")] = f.read_text()
except Exception:
    pass


def big_score(p, label):
    return f"<div style='text-align:center'><div style='font-size:15px;opacity:.7'>{label}</div>" \
           f"<div style='font-size:52px;font-weight:800'>{p*100:.0f}%</div></div>"


def analyse(story, company):
    if not story or len(story.strip()) < 20:
        raise gr.Error("Please describe what happened (at least a couple of sentences).")
    r = predict(story)
    ticked, qs = check_and_questions(story, r["issue"])
    labels = all_labels(r["issue"])
    context = base_rate_text(company, r["issue"])
    return (
        big_score(r["p_win"], "Chance of getting money back"),
        f"**Issue type:** {r['issue']}" + (f"\n\n{context}" if context else ""),
        sentence_impact(story),
        gr.CheckboxGroup(choices=labels, value=ticked),
        "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs)) or "Nothing missing. Nice!",
    )


def improve(story, answers, country):
    if not story or len(story.strip()) < 20:
        raise gr.Error("Check your complaint first.")
    best, old, new, log = rewrite_complaint(story, answers)
    print("rewrite log:", log)
    arrow = f"<div style='text-align:center;font-size:20px'>{old*100:.0f}% &rarr; <b style='font-size:40px'>{new*100:.0f}%</b></div>"
    letters = bank_letter(best) + "\n\n\n" + regulator_version(best, country)
    return arrow, best, letters


def load_demo(name):
    return DEMO_STORIES.get(name, "")


with gr.Blocks(title="Claimback") as demo:
    gr.Markdown("# Claimback\nWronged by your bank? Tell us what happened. We'll show your odds of getting "
                "money back and help you make your case.  \n*Not legal advice. Complaining to the CFPB or AFCA is free.*")
    with gr.Row():
        with gr.Column(scale=3):
            if DEMO_STORIES:
                demo_pick = gr.Dropdown(list(DEMO_STORIES), label="Load a demo story (optional)")
            story = gr.Textbox(lines=10, label="1. What happened?",
                               placeholder="Describe it in your own words: what happened, when, how much, what the bank said...")
            with gr.Row():
                company = gr.Textbox(label="Bank or company (optional)")
                country = gr.Radio(["United States", "Australia"], value="United States", label="Where is your bank?")
            go = gr.Button("Check my complaint", variant="primary")
        with gr.Column(scale=2):
            score = gr.HTML()
            issue = gr.Markdown()
            checklist = gr.CheckboxGroup(label="Evidence checklist", interactive=False)
    highlights = gr.HighlightedText(label="What helps (green) or hurts (red) your case",
                                    color_map={"+": "green", "-": "red"}, show_legend=False)
    gr.Markdown("### 2. Fill the gaps")
    qs = gr.Markdown()
    answers = gr.Textbox(lines=5, label="Your answers (write them in any order)")
    fix = gr.Button("Rewrite my complaint", variant="primary")
    new_score = gr.HTML()
    with gr.Row():
        rewritten = gr.Textbox(lines=16, label="3. Your stronger complaint")
        letters = gr.Textbox(lines=16, label="4. Ready to send")

    go.click(analyse, [story, company], [score, issue, highlights, checklist, qs])
    fix.click(improve, [story, answers, country], [new_score, rewritten, letters])
    if DEMO_STORIES:
        demo_pick.change(load_demo, demo_pick, story)

if __name__ == "__main__":
    print(f"Model backend: {config.MODEL_BACKEND}")
    demo.launch()
