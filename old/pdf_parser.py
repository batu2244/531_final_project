"""
pdf_parser.py

Parses plain-text resumes from the Output_Resumes directory and produces
JSON representations for each person.

Output: parsed_resumes.json  (list of person objects)
"""

import json
import os
import re
from pathlib import Path


RESUMES_DIR = Path(__file__).parent / "Output_Resumes"
OUTPUT_FILE = Path(__file__).parent / "parsed_resumes.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_sections(text: str) -> dict[str, str]:
    """Split resume text into named sections.

    The section headers are ALL-CAPS lines (optionally with spaces).
    Returns a dict mapping header -> raw section body.
    """
    section_pattern = re.compile(r'^([A-Z][A-Z /&]+)\s*$', re.MULTILINE)
    headers = [(m.group(1).strip(), m.start()) for m in section_pattern.finditer(text)]

    sections: dict[str, str] = {}
    for i, (header, start) in enumerate(headers):
        end = headers[i + 1][1] if i + 1 < len(headers) else len(text)
        body = text[start + len(header):end].strip()
        sections[header] = body

    return sections


def parse_technical_skills(body: str) -> dict[str, list[str]]:
    """Parse a skills section into a dict of category -> list of skills."""
    skills: dict[str, list[str]] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            category, _, values = line.partition(':')
            skills[category.strip()] = [v.strip() for v in values.split(',') if v.strip()]
        else:
            skills.setdefault('Other', []).append(line)
    return skills


def parse_experience(body: str) -> list[dict]:
    """Parse professional or project experience into structured entries."""
    entries = []
    lines = [l.rstrip() for l in body.splitlines()]

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        # Detect a job block: a non-bullet, non-empty line followed by
        # "Company | years" or just bullet points (project entries).
        if line.startswith('-'):
            # Standalone bullet (project experience style)
            entries.append({'description': line.lstrip('- ').strip()})
            i += 1
            continue

        # Possible job title
        title = line.strip()
        i += 1

        # Peek ahead for "Company | date_range"
        company = None
        date_range = None
        bullets: list[str] = []

        if i < len(lines):
            next_line = lines[i].strip()
            company_match = re.match(r'^(.+?)\s*[|–-]\s*(.+)$', next_line)
            if company_match:
                company = company_match.group(1).strip()
                date_range = company_match.group(2).strip()
                i += 1

        # Collect bullet points
        while i < len(lines):
            bl = lines[i].strip()
            if not bl:
                i += 1
                break
            if bl.startswith('-'):
                bullets.append(bl.lstrip('- ').strip())
                i += 1
            else:
                break

        entry: dict = {'title': title}
        if company:
            entry['company'] = company
        if date_range:
            entry['date_range'] = date_range
        if bullets:
            entry['responsibilities'] = bullets

        entries.append(entry)

    return entries


def parse_education(body: str) -> list[dict]:
    """Parse education section into degree / institution pairs."""
    entries = []
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        degree = lines[i]
        institution = lines[i + 1] if i + 1 < len(lines) else None
        entry: dict = {'degree': degree}
        if institution and not institution.isupper():
            entry['institution'] = institution
            i += 2
        else:
            i += 1
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_resume(file_path: Path) -> dict:
    text = file_path.read_text(encoding='utf-8')

    # Extract name from first non-empty line
    name = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            name_match = re.match(r'^Name:\s*(.+)$', stripped)
            name = name_match.group(1).strip() if name_match else stripped
            break

    sections = split_sections(text)

    person: dict = {'name': name, 'source_file': str(file_path)}

    if 'TECHNICAL SKILLS' in sections:
        person['technical_skills'] = parse_technical_skills(sections['TECHNICAL SKILLS'])

    if 'PROFESSIONAL EXPERIENCE' in sections:
        person['professional_experience'] = parse_experience(sections['PROFESSIONAL EXPERIENCE'])

    if 'PROJECT EXPERIENCE' in sections:
        person['project_experience'] = parse_experience(sections['PROJECT EXPERIENCE'])

    if 'EDUCATION' in sections:
        person['education'] = parse_education(sections['EDUCATION'])

    return person



def parse_all_resumes(resumes_dir: Path = RESUMES_DIR) -> list[dict]:
    results = []
    for root, dirs, files in os.walk(resumes_dir):
        # Skip hidden dirs/files
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for fname in files:
            if fname.startswith('.'):
                continue
            fpath = Path(root) / fname
            try:
                person = parse_resume(fpath)
                results.append(person)
            except Exception as exc:
                print(f"  [WARN] Could not parse {fpath}: {exc}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"Scanning: {RESUMES_DIR}")
    people = parse_all_resumes()
    print(f"Parsed {len(people)} resume(s).")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(people, f, indent=2, ensure_ascii=False)

    print(f"Output written to: {OUTPUT_FILE}")
