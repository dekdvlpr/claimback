"""Turns the rewritten complaint into ready-to-send text, plus base-rate context."""
import functools
import os

from claimback import config


def bank_letter(complaint):
    return ("To the Complaints Team,\n\n"
            "I am making a formal complaint about the matter below. Please investigate, give me a "
            "written response, and tell me my complaint reference number.\n\n"
            f"{complaint.strip()}\n\n"
            "Kind regards,\n[Your name]\n[Your account number - last 4 digits only]\n[Your phone or email]")


def regulator_version(complaint, country):
    if country == "Australia":
        return ("--- AFCA complaint (afca.org.au, free) ---\n"
                "Before you lodge: AFCA usually expects you to complain to the bank first and give it "
                "time to respond. Have the bank's reference number ready.\n\n"
                "What happened / why you are complaining:\n" + complaint.strip())
    return ("--- CFPB complaint (consumerfinance.gov/complaint, free) ---\n"
            "Paste this into 'What happened?'. The CFPB sends it to the company, which is expected "
            "to respond.\n\n" + complaint.strip())


@functools.lru_cache(maxsize=1)
def _base_rates():
    if not os.path.exists(config.BASE_RATES_PATH):
        return None
    import pandas as pd
    return pd.read_csv(config.BASE_RATES_PATH)


def base_rate_text(company, issue):
    """e.g. 'For Bank X and this issue, 23% of 1,402 CFPB complaints ended with money back.'"""
    df = _base_rates()
    if df is None or not company or not company.strip():
        return ""
    hit = df[df["company"].str.lower().str.contains(company.strip().lower(), regex=False, na=False)
             & (df["issue_group"] == issue)]
    if hit.empty:
        return ""
    row = hit.sort_values("n", ascending=False).iloc[0]
    return (f"For {row['company'].title()} and this issue, {row['rate']*100:.0f}% of "
            f"{int(row['n']):,} CFPB complaints ended with money back.")
