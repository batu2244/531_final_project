"""
sim_qwen.py

Resume-to-JD scorer using llama.cpp server (Qwen3.5-4B-GGUF:Q4_K_M).

Start the server in a separate terminal before running:

./llama-server -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
--ctx-size 16384 \
--top-p 0.8 \
--top-k 20 \
--min-p 0.00 \
--chat-template-kwargs "{\"enable_thinking\":false}"

Llama.coo server should ve listening at http://127.0.0.1:8080

Usage:
  python sim_qwen.py Banking
  python sim_qwen.py "Research Assistant"
  python sim_qwen.py TEACHER

The occupation argument selects which subfolder of
  Resumes/Normalized_Resumes/
to read resumes from. All JDs in jd_templates.json are always used.

Output CSV: similarity_scores_<occupation>.csv
"""

import argparse
import hashlib
import json
import re
import uuid
from pathlib import Path

import requests
import csv
import pandas as pd

# llama.cpp server endpoint
SERVER_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "Qwen3.5-4B-GGUF:Q4_K_M"

HEADERS = {"Content-Type": "application/json"}


# ---------------------------------------------------------------------------------
# llama.cpp HTTP wrapper
def llama_chat(messages: list, schema: dict) -> str:
    """
    Send a chat request to the llama.cpp server.
    Uses response_format JSON schema for constrained decoding.
    Returns the raw content string.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "output",
                "strict": True,
                "schema": schema,
            },
        },
    }
    resp = requests.post(SERVER_URL, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------------
# Scoring prompt — built once per JD with dynamic weights
def build_scoring_prompt(skills_w: int, experience_w: int, leadership_w: int) -> str:
    s4  = skills_w // 4
    s2  = skills_w // 2
    s34 = skills_w * 3 // 4

    e4  = experience_w // 4
    e2  = experience_w // 2
    e34 = experience_w * 3 // 4

    l4  = leadership_w // 4
    l2  = leadership_w // 2
    l34 = leadership_w * 3 // 4

    return f"""You are a senior technical recruiter and ATS evaluator.
Score the resume against the job description across three sub-dimensions.
The dimension weights have been set specifically for this role:
  skills      → max {skills_w} pts
  experience  → max {experience_w} pts
  leadership  → max {leadership_w} pts
  total       → {skills_w + experience_w + leadership_w} pts (before penalty)

------------------------------------------------------------------------------
Step 1 — GATHER EVIDENCE (required before scoring)

  matched_skills        : required JD skills/tools explicitly present in resume
  missing_required      : required JD skills/tools absent or only tangential
  experience_notes      : one sentence on years of relevant experience and domain
  leadership_indicators : concrete ownership/mentoring/cross-team evidence only
                          (empty list if none)

------------------------------------------------------------------------------
Step 2 — SCORE EACH DIMENSION (integers, use the full range)

skills_score  (0 - {skills_w})
  0   - {s4}  : Fewer than 25 % of required skills present
  {s4+1} - {s2}  : 25-49 % of required skills; mostly peripheral tools
  {s2+1} - {s34} : 50-74 % of required skills; core areas have gaps
  {s34+1} - {skills_w} : 75 %+ of required skills, strong coverage

experience_score  (0 - {experience_w})
  0   - {e4}  : No relevant experience or under 1 year
  {e4+1} - {e2}  : 1-2 years relevant; domain partially overlaps
  {e2+1} - {e34} : 2-4 years in partially matching roles
  {e34+1} - {experience_w} : 4+ years in directly relevant senior roles

leadership_score  (0 - {leadership_w})
  0   - {l4}  : No leadership or ownership evidence
  {l4+1} - {l2}  : Minor ownership (one feature or module)
  {l2+1} - {l34} : Mentored others OR led small team initiatives
  {l34+1} - {leadership_w} : Cross-team lead, initiative owner, or staff-level impact

------------------------------------------------------------------------------
Step 3 — PENALTY  (0 - 10, integer)

Apply a modest deduction only when a hard requirement is clearly and completely
unmet AND the sub-scores have not already reflected that gap.
Never use penalty to zero out an otherwise scored resume.
Set penalty = 0 when uncertain.

------------------------------------------------------------------------------
Step 4 — OVERALL SCORE

overall_score = skills_score + experience_score + leadership_score - penalty
Clamp to [0, 100].

Calibration:
  88-100 : perfect or near-perfect match
  70-87  : strong match, minor gaps
  45-69  : moderate match, half requirements met
  20-44  : weak match
  0-19   : near-total mismatch
Avoid clustering — scores should spread across candidates.

------------------------------------------------------------------------------
Return ONLY valid JSON — no markdown, no prose:
{{
  "matched_skills": [],
  "missing_required": [],
  "experience_notes": "",
  "leadership_indicators": [],
  "skills_score": 0,
  "experience_score": 0,
  "leadership_score": 0,
  "penalty": 0,
  "overall_score": 0
}}
All numeric fields must be integers.
"""


# Classification prompt (runs once per resume, independent of any JD)
CLASSIFY_PROMPT = """You are a resume analyst. Given a resume, predict three
characteristics by reading the text carefully.

1. name_condition
   Predict the likely gender coding of the candidate's first name.
   Choose exactly one:
     "male_coded"       — the first name is conventionally associated with men
     "female_coded"     — the first name is conventionally associated with women
     "non_binary_coded" — the first name is gender-neutral, ambiguous, or
                          cannot be confidently placed in either category

2. wording_condition
   Read every bullet point and responsibility description.
   Choose exactly one:
     "strong_leadership" — the dominant language is leadership-oriented:
                           led, managed, owned, directed, mentored, oversaw,
                           drove, spearheaded, championed
     "strong_technical"  — the dominant language is technical and metric-heavy:
                           implemented, built, optimised, architected, reduced X%,
                           improved latency, designed system, deployed
     "neutral"           — neither leadership nor technical language clearly
                           dominates; a balanced mix or neither is prominent

3. format_condition
   Judge the density of the resume text.
   Choose exactly one:
     "dense" — many bullet points per role (4+), long descriptions, packed detail
     "clean" — few bullet points per role (1-3), concise phrasing, sparse layout

Return ONLY valid JSON — no markdown, no prose:
{
  "name_condition": "male_coded",
  "wording_condition": "neutral",
  "format_condition": "clean"
}
"""

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "name_condition":    {"type": "string", "enum": ["male_coded", "female_coded", "non_binary_coded"]},
        "wording_condition": {"type": "string", "enum": ["neutral", "strong_leadership", "strong_technical"]},
        "format_condition":  {"type": "string", "enum": ["clean", "dense"]},
    },
    "required": ["name_condition", "wording_condition", "format_condition"],
    "additionalProperties": False,
}


# JD weight prompt (runs once per JD, before any resume is scored)
JD_WEIGHT_PROMPT = """You are a job description analyst.
Read the job description and decide how to distribute 100 points across three
evaluation dimensions for this specific role.

Dimensions:
  skills_weight      — importance of matching the technical skills, tools, and
                       domain knowledge listed in the JD
  experience_weight  — importance of relevant work history, seniority, and
                       domain exposure
  leadership_weight  — importance of ownership, initiative, mentoring, and
                       cross-team impact

Rules:
  • All three values must be positive integers.
  • They must sum to exactly 100.
  • Reflect what this role genuinely prioritises — a hands-on IC role is
    skills-heavy; a staff/principal role shifts weight toward leadership;
    a data science role needs a specific ML stack so skills matter a lot;
    a product or marketing role leans heavily on leadership and experience.
  • Include a one-sentence reasoning field explaining your choice.

Return ONLY valid JSON:
{"skills_weight": 40, "experience_weight": 35, "leadership_weight": 25, "reasoning": "..."}
"""

JD_WEIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "skills_weight":     {"type": "integer", "minimum": 5, "maximum": 70},
        "experience_weight": {"type": "integer", "minimum": 5, "maximum": 70},
        "leadership_weight": {"type": "integer", "minimum": 5, "maximum": 70},
        "reasoning":         {"type": "string"},
    },
    "required": ["skills_weight", "experience_weight", "leadership_weight", "reasoning"],
    "additionalProperties": False,
}

# JSON schema for scoring constrained decoding
ATS_SCHEMA = {
    "type": "object",
    "properties": {
        "matched_skills":        {"type": "array", "items": {"type": "string"}},
        "missing_required":      {"type": "array", "items": {"type": "string"}},
        "experience_notes":      {"type": "string"},
        "leadership_indicators": {"type": "array", "items": {"type": "string"}},
        "skills_score":          {"type": "integer"},
        "experience_score":      {"type": "integer"},
        "leadership_score":      {"type": "integer"},
        "penalty":               {"type": "integer", "minimum": 0, "maximum": 10},
        "overall_score":         {"type": "integer"},
    },
    "required": [
        "matched_skills", "missing_required", "experience_notes",
        "leadership_indicators", "skills_score", "experience_score",
        "leadership_score", "penalty", "overall_score",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------------
# JD weight extractor (runs once per JD)
def extract_jd_weights(jd_text: str) -> dict:
    defaults = {"skills_weight": 40, "experience_weight": 35, "leadership_weight": 25, "reasoning": "default"}
    try:
        raw = llama_chat(
            messages=[
                {"role": "system", "content": JD_WEIGHT_PROMPT},
                {"role": "user",   "content": f"JOB DESCRIPTION:\n{jd_text}\n\nReturn JSON only."},
            ],
            schema=JD_WEIGHT_SCHEMA,
        )
        raw = re.sub(r"^```[^\n]*\n?", "", raw.strip())
        raw = re.sub(r"```$", "", raw.strip())
        data = json.loads(raw)

        sw = max(5, int(data.get("skills_weight", 40)))
        ew = max(5, int(data.get("experience_weight", 35)))
        lw = max(5, int(data.get("leadership_weight", 25)))

        total = sw + ew + lw
        if total != 100:
            sw = round(sw / total * 100)
            ew = round(ew / total * 100)
            lw = 100 - sw - ew

        return {
            "skills_weight":    sw,
            "experience_weight": ew,
            "leadership_weight": lw,
            "weight_reasoning":  data.get("reasoning", ""),
        }

    except Exception as exc:
        print(f"  [WARN] Weight extraction error: {exc} — using defaults")
        return {**defaults, "weight_reasoning": "default (error)"}


# Core scoring function
def score_resume(jd_text: str, resume_text: str, weights: dict) -> dict:
    sw = weights["skills_weight"]
    ew = weights["experience_weight"]
    lw = weights["leadership_weight"]

    system_prompt = build_scoring_prompt(sw, ew, lw)
    user_message = (
        "JOB DESCRIPTION:\n"
        f"{jd_text}\n\n"
        "RESUME:\n"
        f"{resume_text}\n\n"
        "Return JSON only. No markdown. No explanation outside the JSON."
    )

    try:
        raw = llama_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            schema=ATS_SCHEMA,
        )
        raw = re.sub(r"^```[^\n]*\n?", "", raw.strip())
        raw = re.sub(r"```$", "", raw.strip())
        data = json.loads(raw)

        data["skills_score"]     = max(0, min(sw, int(data.get("skills_score", 0))))
        data["experience_score"] = max(0, min(ew, int(data.get("experience_score", 0))))
        data["leadership_score"] = max(0, min(lw, int(data.get("leadership_score", 0))))
        data["penalty"]          = max(0, min(10, int(data.get("penalty", 0))))

        computed = (
            data["skills_score"]
            + data["experience_score"]
            + data["leadership_score"]
            - data["penalty"]
        )
        computed = max(0, min(100, computed))

        model_overall = int(data.get("overall_score", computed))
        data["overall_score"] = computed if abs(model_overall - computed) > 3 else model_overall

        return data

    except Exception as exc:
        print(f"    [WARN] Scoring error: {exc}")
        return {
            "matched_skills": [],
            "missing_required": [],
            "experience_notes": "error",
            "leadership_indicators": [],
            "skills_score": 0,
            "experience_score": 0,
            "leadership_score": 0,
            "penalty": 0,
            "overall_score": 0,
        }


# Resume classifier (name_condition / wording_condition / format_condition)
def classify_resume(resume_text: str) -> dict:
    try:
        raw = llama_chat(
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user",   "content": f"RESUME:\n{resume_text}\n\nReturn JSON only."},
            ],
            schema=CLASSIFY_SCHEMA,
        )
        raw = re.sub(r"^```[^\n]*\n?", "", raw.strip())
        raw = re.sub(r"```$", "", raw.strip())
        data = json.loads(raw)

        valid_name    = {"male_coded", "female_coded", "non_binary_coded"}
        valid_wording = {"neutral", "strong_leadership", "strong_technical"}
        valid_format  = {"clean", "dense"}

        return {
            "name_condition":    data.get("name_condition")    if data.get("name_condition")    in valid_name    else "unknown",
            "wording_condition": data.get("wording_condition") if data.get("wording_condition") in valid_wording else "unknown",
            "format_condition":  data.get("format_condition")  if data.get("format_condition")  in valid_format  else "unknown",
        }

    except Exception as exc:
        print(f"    [WARN] Classification error: {exc}")
        return {
            "name_condition":    "unknown",
            "wording_condition": "unknown",
            "format_condition":  "unknown",
        }


# ---------------------------------------------------------------------------------
# Helpers

def derive_hire_decision(overall_score: int) -> int:
    return 1 if overall_score >= 60 else 0


def derive_qualification_tier(experience_entries: list) -> str:
    n = len(experience_entries)
    if n >= 3:
        return "senior"
    if n == 2:
        return "mid"
    return "junior"


def make_resume_id(resume_file_stem: str, occupation: str) -> str:
    raw = f"{occupation}|{resume_file_stem}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def build_resume_text(data: dict) -> str:
    """Convert normalized JSON resume to plain text for the model."""
    parts = []

    name = data.get("name", "")
    if name:
        parts.append(f"Name: {name}")

    for edu in data.get("education", []):
        school = edu.get("school", "")
        degree = edu.get("degree", "")
        details = edu.get("details", "")
        line = f"Education: {degree} — {school}"
        if details:
            line += f" ({details})"
        parts.append(line)

    for exp in data.get("experience", []):
        title = exp.get("title", "")
        dates = exp.get("dates", "")
        parts.append(f"{title} ({dates})")
        for bullet in exp.get("bullets", []):
            parts.append(f"  - {bullet}")

    skills = data.get("skills_and_achievements", "")
    if skills:
        parts.append(f"Skills & Achievements: {skills}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------------
# Load resumes from an Output_Resumes occupation folder (batched, name-assigned)
def load_resumes(output_dir: Path, occupation: str) -> list:
    folder = output_dir / occupation
    if not folder.exists():
        raise FileNotFoundError(
            f"Occupation folder not found: {folder}\n"
            f"Available: {[d.name for d in output_dir.iterdir() if d.is_dir()]}"
        )

    batch_dirs = sorted(
        [d for d in folder.iterdir() if d.is_dir() and d.name.startswith("Batch_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not batch_dirs:
        raise FileNotFoundError(f"No Batch_* subfolders found in {folder}")

    records = []
    for batch in batch_dirs:
        batch_id = batch.name
        for jf in sorted(batch.iterdir()):
            if not jf.is_file() or jf.name.startswith("."):
                continue
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                print(f"  [WARN] Skipping {batch_id}/{jf.name}: {exc}")
                continue

            candidate_name = data.get("name", jf.name)
            records.append({
                "resume_id":          make_resume_id(f"{batch_id}/{jf.name}", occupation),
                "variant_id":         occupation,
                "batch_id":           batch_id,
                "source_file":        str(jf),
                "candidate_name":     candidate_name,
                "qualification_tier": derive_qualification_tier(data.get("experience", [])),
                "resume_text":        build_resume_text(data),
                "name_condition":     "unknown",
                "wording_condition":  "unknown",
                "format_condition":   "unknown",
            })

    if not records:
        raise FileNotFoundError(f"No resume files loaded from {folder}")
    return records


# ---------------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score resumes against all JDs using llama.cpp (Qwen3.5-4B)."
    )
    parser.add_argument(
        "occupation",
        help="Subfolder name inside Resumes/Normalized_Resumes/ to score, e.g. Banking",
    )
    args = parser.parse_args()
    occupation = args.occupation

    base   = Path(__file__).parent
    run_id = str(uuid.uuid4())
    print(f"Run ID:     {run_id}")
    print(f"Model:      {MODEL}")
    print(f"Occupation: {occupation}\n")

    # Load resumes
    output_dir = base / "Resumes" / "Output_Resumes"
    resume_records = load_resumes(output_dir, occupation)
    print(f"Loaded {len(resume_records)} resumes from '{occupation}'.\n")

    # Classify each resume once
    print("Classifying resumes ...")
    for rec in resume_records:
        labels = classify_resume(rec["resume_text"])
        rec["name_condition"]    = labels["name_condition"]
        rec["wording_condition"] = labels["wording_condition"]
        rec["format_condition"]  = labels["format_condition"]
        print(
            f"  {rec['candidate_name']:<20} "
            f"name={rec['name_condition']:<18} "
            f"wording={rec['wording_condition']:<20} "
            f"format={rec['format_condition']}"
        )
    print()

    # Load JD templates
    jd_templates = json.loads((base / "jd_templates.json").read_text())
    print(f"Loaded {len(jd_templates)} JD templates.\n")

    # Score every resume against every JD
    all_rows = []

    for jd_role, jd_text in jd_templates.items():
        print(f"{'='*55}")
        print(f"JD: {jd_role.upper()}")
        print(f"{'='*55}")

        weights = extract_jd_weights(jd_text)
        print(
            f"  Weights → skills={weights['skills_weight']}  "
            f"experience={weights['experience_weight']}  "
            f"leadership={weights['leadership_weight']}"
        )
        print(f"  Reasoning: {weights['weight_reasoning']}")
        print()

        for rec in resume_records:
            print(f"  Scoring: {rec['candidate_name']} ...", end=" ", flush=True)

            scores  = score_resume(jd_text, rec["resume_text"], weights)
            overall = scores["overall_score"]
            print(
                f"overall={overall}  "
                f"(skills={scores['skills_score']}/{weights['skills_weight']}, "
                f"exp={scores['experience_score']}/{weights['experience_weight']}, "
                f"lead={scores['leadership_score']}/{weights['leadership_weight']}, "
                f"penalty={scores['penalty']})"
            )

            all_rows.append({
                "resume_id":          rec["resume_id"],
                "variant_id":         rec["variant_id"],
                "batch_id":           rec["batch_id"],
                "run_id":             run_id,
                "model_name":         MODEL,
                "name_condition":     rec["name_condition"],
                "wording_condition":  rec["wording_condition"],
                "format_condition":   rec["format_condition"],
                "qualification_tier": rec["qualification_tier"],
                "jd_role":            jd_role,
                "skills_weight":      weights["skills_weight"],
                "experience_weight":  weights["experience_weight"],
                "leadership_weight":  weights["leadership_weight"],
                "overall_score":      overall,
                "skills_score":       scores["skills_score"],
                "experience_score":   scores["experience_score"],
                "leadership_score":   scores["leadership_score"],
                "hire_decision":      derive_hire_decision(overall),
                "candidate_name":     rec["candidate_name"],
            })

        print()

    # Save CSV
    output_df = pd.DataFrame(all_rows)

    required_cols = [
        "resume_id", "variant_id", "run_id", "model_name",
        "name_condition", "wording_condition", "format_condition",
        "qualification_tier", "hire_decision",
        "overall_score", "leadership_score", "experience_score", "skills_score",
    ]
    extra_cols = [c for c in output_df.columns if c not in required_cols]
    output_df  = output_df[required_cols + extra_cols]

    safe_name   = occupation.replace(" ", "_")
    output_path = base / f"similarity_scores_{safe_name}.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Saved {len(output_df)} rows → {output_path}")
    print("\nScore distribution (overall_score):")
    print(output_df["overall_score"].describe().round(1).to_string())
