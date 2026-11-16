"""Evidence checklist + gap questions.

check(story, issue)     -> list of checklist items the story already covers
questions(story, issue) -> list of questions for the items it is missing
"""
import re

from claimback import llm

# item_key: (what it means, the question to ask if it's missing, offline keyword rule)
CORE = {
    "dates": ("Dates of what happened", "When did this happen? Give the key dates.",
              r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\d{1,2}/\d{1,2}|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday"),
    "amounts": ("Exact amounts", "How much money is involved, exactly?", r"\$\s?\d|\d+\s?dollars"),
    "contacted_company": ("What you already tried with the company",
                          "Have you contacted the company? When, and what did they say?",
                          r"called|contacted|emailed|complain|told the bank|spoke"),
    "reference_number": ("Reference or case number", "Do you have a reference or case number?",
                         r"reference|case (number|#|no)|ticket|claim (number|#)"),
    "remedy_wanted": ("The outcome you want", "What exactly do you want them to do (for example refund $X)?",
                      r"want (a|my|the)? ?(refund|money)|asking for|reimburse|refund"),
}
EXTRA = {
    "Scam or unauthorised payment": {
        "authorised_or_not": ("Whether you made the payment yourself",
                              "Did you make the payment yourself (tricked), or did someone else make it without you?",
                              r"authori[sz]|i (sent|transferred)|without my (permission|knowledge)"),
        "reported_when": ("When you reported it", "When did you first report it to your bank, and how?",
                          r"reported|same day|immediately|called the bank|fraud (team|department)"),
        "shared_codes": ("Whether you shared codes or passwords",
                         "Did you share any one-time code, PIN or password with anyone?",
                         r"\bcode\b|\bpin\b|password|otp"),
        "warnings": ("Bank warnings, or how unusual the payment was",
                     "Did the bank warn you or block anything? Was this payment unusual for you?",
                     r"warn|unusual|never (made|sent)|first time|block"),
    },
    "Fees and charges": {
        "fee_disclosed": ("What the fee was and whether it was disclosed",
                          "What was the fee for, and were you told about it beforehand?",
                          r"fee|charge|overdraft|disclos|told"),
    },
    "Card or transaction dispute": {
        "dispute_details": ("What went wrong and when you disputed it",
                            "When did you dispute it, and did the merchant respond?",
                            r"dispute|merchant|never (arrived|received)|charged twice"),
    },
    "Debt collection": {
        "debt_details": ("Whether you owe the debt and what the collector did",
                         "Do you recognise the debt? What did the collector say or do?",
                         r"(don't|do not) owe|not my debt|collector (called|said)|verify|validation"),
    },
    "Account problems": {
        "account_details": ("What happened to the account and the impact",
                            "Were funds frozen or the account closed? For how long, and how did it affect you?",
                            r"frozen|closed|locked|restricted|weeks|days"),
    },
    "Loan or mortgage servicing": {
        "loan_details": ("What was mishandled", "Which payment or request was mishandled, and when?",
                         r"payment|escrow|modification|statement|late"),
    },
}


def _items(issue):
    return {**CORE, **EXTRA.get(issue, {})}


def _which_present(story, issue):
    items = _items(issue)
    schema = {"type": "object", "additionalProperties": False, "required": list(items),
              "properties": {k: {"type": "boolean"} for k in items}}
    listing = "\n".join(f"- {k}: {v[0]}" for k, v in items.items())
    got = llm.ask(
        "For each item below, answer true if the complaint clearly states it, otherwise false.\n"
        f"{listing}\n\nCOMPLAINT:\n{story}",
        json_schema=schema, max_tokens=2000)
    if got is None:                     # offline rule: keyword match
        t = story.lower()
        got = {k: bool(re.search(v[2], t)) for k, v in items.items()}
    return items, got


def check(story, issue):
    """Labels of checklist items the story already covers (for the ticked boxes)."""
    items, got = _which_present(story, issue)
    return [items[k][0] for k, v in got.items() if v]


def all_labels(issue):
    return [v[0] for v in _items(issue).values()]


def questions(story, issue):
    """Questions for the items that are missing."""
    items, got = _which_present(story, issue)
    return [items[k][1] for k, v in got.items() if not v]


def check_and_questions(story, issue):
    """Both at once with a single LLM call (what the app uses)."""
    items, got = _which_present(story, issue)
    ticked = [items[k][0] for k, v in got.items() if v]
    qs = [items[k][1] for k, v in got.items() if not v]
    return ticked, qs
