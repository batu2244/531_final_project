# ai-ats

Resume-to-JD scoring pipeline using a local llama.cpp server (Qwen3.5-4B).
Scores synthetic resumes against job descriptions and records demographic
condition labels for bias analysis.

## Layout

```
ai-ats/
├── jd_templates.json              # Job descriptions (7 roles)
├── sim_qwen.py                    # Scoring pipeline
├── Resumes/
│   ├── Synthetic_Resumes/         # Raw source resumes
│   ├── Normalized_Resumes/        # Cleaned JSON resumes
│   ├── Output_Resumes/            # Name-assigned, batched resumes (scorer input)
│   ├── resume_formatting.py
│   ├── resume_normalization.py
│   └── name_assignment.py
└── similarity_scores_<occ>.csv    # Output
```

## Pipeline

1. `resume_formatting.py` / `resume_normalization.py` → produce `Normalized_Resumes/`.
2. `name_assignment.py` → samples resumes and assigns demographic-coded names, writes `Output_Resumes/<OCC>/Batch_N/`.
3. `sim_qwen.py` → classifies each resume (name/wording/format) and scores it against every JD in `jd_templates.json`.

## Requirements

- Python 3.10+
- `pip install requests pandas`
- llama.cpp with a Qwen3.5-4B-GGUF model

## Running

### 1. Start the llama.cpp server (separate terminal, leave running)

```bash
./llama-server -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
  --ctx-size 16384 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.00 \
  --chat-template-kwargs '{"enable_thinking":false}'
```

Must listen on `http://127.0.0.1:8080`.

### 2. Assign names (once per occupation, if not done already)

```bash
python Resumes/name_assignment.py ACCOUNTANT 20
```

- Arg 1: occupation folder name (must exist in `Normalized_Resumes/`).
- Arg 2: number of batches. Each batch = 10 resumes (5 races × 2 genders).

### 3. Score resumes against all JDs

```bash
python sim_qwen.py ACCOUNTANT
```

Available occupations: `ACCOUNTANT`, `Apparel`, `Banking`, `Finance`,
`Research_Assistant`, `TEACHER` (case-sensitive, underscore not space).

Output: `similarity_scores_<OCC>.csv` in the project root.

## Output columns

`resume_id, variant_id, batch_id, run_id, model_name, name_condition,
wording_condition, format_condition, qualification_tier, jd_role,
skills_weight, experience_weight, leadership_weight, overall_score,
skills_score, experience_score, leadership_score, hire_decision,
candidate_name`

## Runtime

~1,400 server calls per folder (200 resumes × 7 JDs + classify + weights).
On an Apple M5 with Qwen3.5-4B Q4_K_M, expect ~60 min per folder.
