
"""Safely rewrites consumer complaints without inventing facts."""

import re

from claimback import config, llm
from claimback.predict import predict


SYSTEM = """You are an assistant that improves consumer complaints against
banks, card issuers, lenders and debt collectors.

Your task is to improve clarity, organisation and readability.

STRICT RULES:
- Use ONLY facts explicitly provided in ORIGINAL and ANSWERS.
- Never invent dates, amounts, names, events, evidence or feelings.
- Never assume the company acted unlawfully or was negligent.
- Never invent legal rights, laws or regulations.
- Never invent what the customer has already done.
- Never invent what outcome the customer wants.
- If information is missing, leave it out.
- Do not use placeholders.
- Write in the first person, as the customer.
- Preserve the meaning of the original complaint.
- Do not exaggerate the seriousness of the complaint.
- Use plain, calm and professional English.
- Remove unnecessary repetition.

Use these four headings:

1. What happened
2. What I have already done and how the company responded
3. Why I believe this issue should be resolved
4. What I am asking for

Only include information supported by ORIGINAL and ANSWERS.
If a section has insufficient information, do not invent content for it.

Return ONLY the rewritten complaint.
"""


def numbers_in(text):
    """Extract numbers, amounts and dates for a basic factual safeguard."""
    found = set()

    for number in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        number = number.replace(",", "")

        if "." in number:
            number = number.rstrip("0").rstrip(".")

        found.add(number)

    return found


def numbers_guard_ok(rewrite, source):
    allowed = numbers_in(source) | {"1", "2", "3", "4"}
    return not (numbers_in(rewrite) - allowed)


def claims_guard_ok(rewrite, source):
    """Reject a rewrite if Gemini cannot verify its factual claims."""

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["unsupported_claims"],
        "properties": {
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }

    got = llm.ask(
        "Compare REWRITE against SOURCE.\n"
        "Identify every factual statement in REWRITE that is not "
        "supported by SOURCE.\n"
        "Do not flag harmless rewording.\n"
        "Return an empty unsupported_claims array only if all "
        "factual statements are supported.\n\n"
        f"SOURCE:\n{source}\n\n"
        f"REWRITE:\n{rewrite}",
        json_schema=schema,
        max_tokens=2000
    )

    # IMPORTANT: An API failure must never count as a successful check.
    if not isinstance(got, dict):
        return False

    unsupported = got.get("unsupported_claims")

    if not isinstance(unsupported, list):
        return False

    return len(unsupported) == 0


def rewrite_complaint(original, answers=""):
    """
    Returns:
        best_text, old_score, new_score, log
    """

    source = original.strip() + "\n" + (answers or "").strip()
    old = predict(original)["p_win"]

    # Do not pretend an offline rewrite is AI-generated.
    if not llm.available():
        return (
            original,
            old,
            old,
            ["AI unavailable: original complaint retained."]
        )

    log = []

    # Limit attempts to reduce Gemini API usage.
    attempts = min(config.REWRITE_ATTEMPTS, 2)

    for i in range(attempts):

        cand = llm.ask(
            f"ORIGINAL:\n{original}\n\n"
            f"ANSWERS:\n{answers or '(none provided)'}",
            system=SYSTEM,
            max_tokens=2500
        )

        # Stop immediately if the API fails or quota is exhausted.
        if not cand:
            log.append(f"Attempt {i+1}: Gemini request failed.")
            break

        cand = cand.strip()

        if not cand:
            log.append(f"Attempt {i+1}: empty response.")
            continue

        if not numbers_guard_ok(cand, source):
            log.append(
                f"Attempt {i+1}: rejected, unsupported numbers."
            )
            continue

        if not claims_guard_ok(cand, source):
            log.append(
                f"Attempt {i+1}: rejected, factual verification failed."
            )
            continue

        # A verified rewrite is useful even if the prediction score
        # does not increase. Do not distort the complaint to chase a score.
        new = predict(cand)["p_win"]

        log.append(
            f"Attempt {i+1}: accepted, predicted score {new:.2f}."
        )

        return cand, old, new, log

    # No verified rewrite was produced.
    return original, old, old, log
