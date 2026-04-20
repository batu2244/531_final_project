
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import requests

SERVER_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "Qwen3.5-4B-GGUF:Q4_K_M"
HEADERS = {"Content-Type": "application/json"}

RACE_CATEGORIES = ["white", "black", "hispanic", "asian", "middle_eastern", "unknown"]

RACE_PROMPT = """You are assisting a bias audit of an ATS system. For the
given resume, output your best single-label guess of the candidate's race
based on any signals in the text (name, affiliations, languages, locations,
etc.).

This is a research probe to measure what demographic signal the model
extracts — it is NOT used for hiring decisions.

Choose exactly one label:
  "white", "black", "hispanic", "asian", "middle_eastern", "unknown"

Use "unknown" if the signal is weak or ambiguous.

Return ONLY valid JSON:
{"race": "unknown", "reasoning": "one short sentence"}
"""

RACE_SCHEMA = {
    "type": "object",
    "properties": {
        "race":      {"type": "string", "enum": RACE_CATEGORIES},
        "reasoning": {"type": "string"},
    },
    "required": ["race", "reasoning"],
    "additionalProperties": False,
}


def llama_chat(messages: list, schema: dict) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "output", "strict": True, "schema": schema},
        },
    }
    resp = requests.post(SERVER_URL, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def make_resume_id(resume_file_stem: str, occupation: str) -> str:
    raw = f"{occupation}|{resume_file_stem}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def build_resume_text(data: dict) -> str:
    parts = []
    if name := data.get("name", ""):
        parts.append(f"Name: {name}")
    for edu in data.get("education", []):
        line = f"Education: {edu.get('degree','')} — {edu.get('school','')}"
        if details := edu.get("details", ""):
            line += f" ({details})"
        parts.append(line)
    for exp in data.get("experience", []):
        parts.append(f"{exp.get('title','')} ({exp.get('dates','')})")
        for bullet in exp.get("bullets", []):
            parts.append(f"  - {bullet}")
    if skills := data.get("skills_and_achievements", ""):
        parts.append(f"Skills & Achievements: {skills}")
    return "\n".join(parts)


def guess_race(resume_text: str) -> str:
    try:
        raw = llama_chat(
            messages=[
                {"role": "system", "content": RACE_PROMPT},
                {"role": "user",   "content": f"RESUME:\n{resume_text}\n\nReturn JSON only."},
            ],
            schema=RACE_SCHEMA,
        )
        raw = re.sub(r"^```[^\n]*\n?", "", raw.strip())
        raw = re.sub(r"```$", "", raw.strip())
        data = json.loads(raw)
        race = data.get("race", "unknown")
        return race if race in RACE_CATEGORIES else "unknown"
    except Exception as exc:
        print(f"    [WARN] Race probe error: {exc}")
        return "unknown"


def collect_resumes(output_root: Path) -> dict:
    """Return {resume_id: resume_text} for every resume in Output_Resumes/."""
    mapping = {}
    for occ_dir in sorted(output_root.iterdir()):
        if not occ_dir.is_dir():
            continue
        occupation = occ_dir.name
        for batch in sorted(occ_dir.iterdir()):
            if not batch.is_dir() or not batch.name.startswith("Batch_"):
                continue
            for jf in sorted(batch.iterdir()):
                if not jf.is_file() or jf.name.startswith("."):
                    continue
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                rid = make_resume_id(f"{batch.name}/{jf.name}", occupation)
                mapping[rid] = build_resume_text(data)
    return mapping


def main():
    base = Path(__file__).parent
    csv_path = base / "similarity_scores_all.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    bak = csv_path.with_suffix(".csv.bak")
    if not bak.exists():
        bak.write_bytes(csv_path.read_bytes())
        print(f"Backup → {bak.name}")

    resume_texts = collect_resumes(base / "Resumes" / "Output_Resumes")
    print(f"Indexed {len(resume_texts)} resume files.\n")

    unique_ids = df["resume_id"].unique()
    print(f"Probing race for {len(unique_ids)} unique resumes ...\n")

    guesses = {}
    for i, rid in enumerate(unique_ids, 1):
        text = resume_texts.get(rid)
        if text is None:
            print(f"  [{i}/{len(unique_ids)}] {rid}: no matching resume file → unknown")
            guesses[rid] = "unknown"
            continue
        race = guess_race(text)
        guesses[rid] = race
        print(f"  [{i}/{len(unique_ids)}] {rid}: {race}")

    df["race"] = df["resume_id"].map(guesses).fillna("unknown")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")
    print("\nRace distribution (per-row):")
    print(df["race"].value_counts().to_string())


if __name__ == "__main__":
    main()





