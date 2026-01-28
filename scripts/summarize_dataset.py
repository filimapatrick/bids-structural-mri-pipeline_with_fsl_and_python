#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Use the actual BIDS dataset root
BIDS_ROOT = PROJECT_ROOT / "ds000246_mri"

# Derivatives for this pipeline live under ds000246_mri/derivatives/mri_pipeline
DERIVATIVES_DIR = BIDS_ROOT / "derivatives" / "mri_pipeline"
METRICS_DIR = DERIVATIVES_DIR / "metrics"
SUMMARY_DIR = DERIVATIVES_DIR / "summary"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

PARTICIPANTS_FILE = BIDS_ROOT / "participants.tsv"
OUTPUT_CSV = SUMMARY_DIR / "dataset_summary.csv"

# -----------------------------
# Load participants.tsv
# -----------------------------
if not PARTICIPANTS_FILE.exists():
    raise FileNotFoundError(f"participants.tsv not found at {PARTICIPANTS_FILE}. Expected under ds000246_mri.")

participants = pd.read_csv(PARTICIPANTS_FILE, sep="\t")

rows = []

# -----------------------------
# Iterate over subjects
# -----------------------------
for _, row in participants.iterrows():
    sub_id = row["participant_id"]

    # Current pipeline outputs FAST partial volumes and a per-subject metrics TSV.
    # Summarize based on metrics TSV presence and values.
    metrics_tsv = METRICS_DIR / f"{sub_id}_tissue_volumes.tsv"

    subject_data = {
        "participant_id": sub_id,
        "age": row.get("age", "n/a"),
        "sex": row.get("sex", "n/a"),
        "dominant_hand": row.get("dominant_hand", "n/a"),

        # Outputs
        "metrics_exists": metrics_tsv.exists(),
    }

    # If metrics exist, parse CSF/GM/WM volumes
    if subject_data["metrics_exists"]:
        try:
            mdf = pd.read_csv(metrics_tsv, sep="\t")
            # Expect columns: subject_id, CSF, GM, WM
            subject_data["CSF"] = float(mdf.iloc[0].get("CSF", float("nan")))
            subject_data["GM"] = float(mdf.iloc[0].get("GM", float("nan")))
            subject_data["WM"] = float(mdf.iloc[0].get("WM", float("nan")))
        except Exception:
            subject_data["CSF"] = "n/a"
            subject_data["GM"] = "n/a"
            subject_data["WM"] = "n/a"

    # Simple QC rule: metrics TSV present
    subject_data["qc_pass"] = subject_data["metrics_exists"]

    rows.append(subject_data)

# -----------------------------
# Save summary CSV
# -----------------------------
summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Dataset summary saved to: {OUTPUT_CSV}")
print(f"📊 Subjects processed: {len(summary_df)}")
print(f"✔ QC passed: {summary_df['qc_pass'].sum()}")
