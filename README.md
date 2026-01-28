# Structural MRI Pipeline (Reproducible Guide)

## Setup

1. Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run the standardization script from the project root:

```bash
source .venv/bin/activate
python scripts/standardize.py
```

Or without activating first:

```bash
./.venv/bin/python scripts/standardize.py
```

## Notes

- Dependencies are tracked in `requirements.txt`. Add packages as needed and re-run the install command.
- The script resolves paths relative to the project root, so run commands from the workspace root directory.

---

## Project Layout

- ds000246_mri/ — BIDS-like dataset root
	- participants.tsv — participants table
	- sub-0001/ — subject with T1w
	- sub-emptyroom/ — empty-room recording (no T1w)

- ds000246_mri/derivatives/mri_pipeline/
	- structural_pipeline/ — Nipype working dirs for nodes
	- metrics/ — per-subject metrics TSVs (e.g., sub-0001_tissue_volumes.tsv)
	- summary/ — dataset-level CSV summary

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./.venv/bin/python -c "import nipype, pandas, nibabel; print('Deps OK')"
```

## Running the Pipeline

```bash
./.venv/bin/python scripts/structural_pipeline.py
```

## Summarizing the Dataset

```bash
./.venv/bin/python scripts/summarize_dataset.py
open ds000246_mri/derivatives/mri_pipeline/summary/dataset_summary.csv
```
# bids-structural-mri-pipeline_with_fsl_and_python
