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
