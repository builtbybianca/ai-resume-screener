# AI Resume Screener

A Python tool that screens and scores resumes against a job description using the Claude API.
Designed, built, and piloted in a live HR environment at a global cybersecurity software company.

## What It Does

1. Takes a job description + a folder of resumes (PDF or text)
2. Sends each resume to Claude with a structured scoring rubric
3. Returns a ranked scorecard: overall score, per-criterion breakdown, and a
   plain-language rationale for every score
4. Flags resumes that need human review rather than auto-rejecting anything

See `tests/` for the evaluation suite, including a name-bias invariance check that scores
an identical resume under different names and fails if the scores diverge.

## Why It's Built This Way

**Scoring rubrics, not vibes.** The prompt forces Claude to score against explicit,
job-relevant criteria with stated reasoning — making every decision reviewable.

**Human-in-the-loop by design.** The tool ranks and explains; humans decide.
No resume is rejected by the system. This was a deliberate governance choice made
with HR leadership and legal review in mind.

**Bias guardrails in the prompt.** The scoring prompt explicitly instructs the model
to ignore name, address, school prestige, and employment gaps, and to score only
against the stated criteria.

## Architecture

Job Description ──┐
├──► Prompt Builder ──► Claude API ──► JSON Scorecard ──► Ranked CSV Report
Resume Files ─────┘

## Setup

1. Clone or download this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Anthropic API key
4. Drop resumes in `data/resumes/` and your job description in `data/job_description.txt`
5. Run: `python src/screener.py`

## Scope

This tool scores **fit** — how well a candidate matches the role. It does not verify
the truthfulness of a resume's claims, and it does not screen identity documents. Those
are deliberately kept as separate tools.

## Status

Piloted in a production HR environment. This public version uses synthetic example data;
no candidate information from the pilot is included in this repository.
