from pathlib import Path
import pandas as pd

OCCUPATIONS = ["ACCOUNTANT", "Apparel", "Banking", "Finance", "Research_Assistant", "TEACHER"]

base = Path(__file__).parent
frames = []
for occ in OCCUPATIONS:
    path = base / f"similarity_scores_{occ}.csv"
    df = pd.read_csv(path)
    df["occupation"] = occ
    frames.append(df)

merged = pd.concat(frames, ignore_index=True)
out_path = base / "similarity_scores_all.csv"
merged.to_csv(out_path, index=False)
print(f"Merged {len(merged)} rows from {len(frames)} files → {out_path}")
