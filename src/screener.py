"""
AI Resume Screener
Scores resumes against a job description using the Claude API with a
structured rubric and human-in-the-loop design.

This version separates the model call from the surrounding logic so the
parsing, validation, and ranking can be tested deterministically without
spending API credits. See tests/test_screener.py.

Usage:
    python src/screener.py
"""

import os
import json
import csv
from pathlib import Path

from dotenv import load_dotenv
import anthropic

# Load the API key from the local .env file (never committed to GitHub)
load_dotenv()

# Check https://docs.claude.com for current model names
MODEL = "claude-sonnet-4-20250514"

# The fields every scorecard must contain, with their expected types.
SCORECARD_SCHEMA = {
    "overall_score": int,
    "skills_match": int,
    "experience_relevance": int,
    "rationale": str,
    "human_review_flag": str,
}

SCORING_PROMPT = """You are an experienced, fair technical recruiter scoring a resume
against a job description.

Score ONLY against the stated criteria. Explicitly ignore: candidate name, address,
school prestige, employment gaps, and any demographic signal.

Job description:
{job_description}

Resume:
{resume_text}

Respond with ONLY a JSON object, no other text:
{{
  "overall_score": <0-100>,
  "skills_match": <0-100>,
  "experience_relevance": <0-100>,
  "rationale": "<2-3 sentences explaining the scores>",
  "human_review_flag": "<empty string, or reason a human should look closer>"
}}"""


# ---------------------------------------------------------------------------
# Pure functions (no network) -- these are what the test suite exercises.
# ---------------------------------------------------------------------------

def parse_scorecard(raw: str) -> dict:
    """Turn the model's raw text reply into a scorecard dict.

    Tolerates the model wrapping its JSON in ```json fences or padding it
    with whitespace -- a common real-world failure mode worth handling once
    here rather than at every call site.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Drop a leading ```json or ``` and any trailing ```
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        cleaned = cleaned.removeprefix("json").strip()
        cleaned = cleaned.removesuffix("```").strip()
    return json.loads(cleaned)


def validate_scorecard(card: dict) -> list[str]:
    """Return a list of problems with a scorecard. Empty list means valid.

    Validating the shape of model output before trusting it is the
    difference between a demo and something you'd let near a hiring workflow.
    """
    problems = []
    for field, expected_type in SCORECARD_SCHEMA.items():
        if field not in card:
            problems.append(f"missing field: {field}")
            continue
        if not isinstance(card[field], expected_type):
            problems.append(
                f"{field} should be {expected_type.__name__}, "
                f"got {type(card[field]).__name__}"
            )
    for score_field in ("overall_score", "skills_match", "experience_relevance"):
        value = card.get(score_field)
        if isinstance(value, int) and not (0 <= value <= 100):
            problems.append(f"{score_field} out of range 0-100: {value}")
    return problems


def rank_results(results: list[dict]) -> list[dict]:
    """Sort scorecards by overall score, highest first. Missing scores sink."""
    return sorted(results, key=lambda c: c.get("overall_score", -1), reverse=True)


# ---------------------------------------------------------------------------
# Model call -- isolated so it can be swapped for a mock in tests.
# ---------------------------------------------------------------------------

def score_resume(job_description: str, resume_text: str, client=None) -> dict:
    """Send one resume to Claude and return the parsed, validated scorecard.

    `client` is injectable so tests can pass a fake. In normal use it
    defaults to a real Anthropic client that reads ANTHROPIC_API_KEY.
    """
    if client is None:
        client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": SCORING_PROMPT.format(
                job_description=job_description,
                resume_text=resume_text[:15000],  # safety cap on long resumes
            ),
        }],
    )
    card = parse_scorecard(response.content[0].text)

    problems = validate_scorecard(card)
    if problems:
        # Don't silently trust malformed output -- surface it for review.
        card["human_review_flag"] = (
            (card.get("human_review_flag", "") + " | ").lstrip(" |")
            + "malformed model output: " + "; ".join(problems)
        ).strip(" |")
    return card


def read_resume(path: Path) -> str:
    """Read a resume file (txt or pdf) into plain text."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def main():
    jd_path = Path("data/job_description.txt")
    resumes_dir = Path("data/resumes")
    output_path = Path("scorecard.csv")

    if not jd_path.exists() or not resumes_dir.exists():
        print("Setup needed: add data/job_description.txt and data/resumes/")
        return

    job_description = jd_path.read_text(encoding="utf-8")
    results = []

    for resume_file in sorted(resumes_dir.iterdir()):
        if resume_file.suffix.lower() not in {".txt", ".pdf"}:
            continue
        print(f"Scoring {resume_file.name}...")
        try:
            card = score_resume(job_description, read_resume(resume_file))
            card["candidate"] = resume_file.stem
            results.append(card)
        except Exception as e:
            print(f"  Skipped ({e})")

    results = rank_results(results)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "candidate", "overall_score", "skills_match",
            "experience_relevance", "human_review_flag", "rationale",
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Ranked scorecard written to {output_path}")
    print("Reminder: this tool ranks and explains. Humans decide.")


if __name__ == "__main__":
    main()
