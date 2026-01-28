from pathlib import Path
from nipype import Node, Workflow, config, logging
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.fsl import BET, FAST
from nipype.interfaces.freesurfer import ReconAll
import os
import shutil
import numpy as np
import nibabel as nib
import pandas as pd

# -----------------------------------------------------
# Nipype configuration (logging & crash files)
# -----------------------------------------------------
config.enable_debug_mode()
config.update_config({
    "logging": {
        "workflow_level": "INFO",
        "interface_level": "INFO"
    },
    "execution": {
        "crashfile_format": "txt"
    }
})

log = logging.getLogger("nipype.workflow")

# -----------------------------------------------------
# Paths
# -----------------------------------------------------
bids_root = Path("ds000246_mri")
derivatives_dir = bids_root / "derivatives"

pipeline_dir = derivatives_dir / "mri_pipeline"
pipeline_dir.mkdir(parents=True, exist_ok=True)

freesurfer_dir = derivatives_dir / "freesurfer"
freesurfer_dir.mkdir(parents=True, exist_ok=True)

# FreeSurfer environment
os.environ["SUBJECTS_DIR"] = str(freesurfer_dir)

# -----------------------------------------------------
# Subject discovery (BIDS)
# -----------------------------------------------------
all_subjects = sorted([p.name for p in bids_root.glob("sub-*") if p.is_dir()])
subjects = []
for sid in all_subjects:
    t1_candidate = bids_root / sid / "anat" / f"{sid}_T1w.nii.gz"
    if t1_candidate.exists():
        subjects.append(sid)

if not subjects:
    raise RuntimeError("No BIDS subjects with T1w images found")

log.info(f"Discovered {len(all_subjects)} subjects: {all_subjects}")
log.info(f"Using {len(subjects)} subjects with T1w: {subjects}")

# -----------------------------------------------------
# Workflow
# -----------------------------------------------------
wf = Workflow(
    name="structural_pipeline",
    base_dir=str(pipeline_dir)
)

# -----------------------------------------------------
# Input node (multi-subject)
# -----------------------------------------------------
inputnode = Node(
    IdentityInterface(fields=["subject_id"]),
    name="inputnode"
)

inputnode.iterables = [
    ("subject_id", subjects),
]

# -----------------------------------------------------
# Resolve T1w path per subject
# -----------------------------------------------------
def get_t1w_path(subject_id, bids_root):
    from pathlib import Path
    t1w = Path(bids_root) / subject_id / "anat" / f"{subject_id}_T1w.nii.gz"
    if not t1w.exists():
        raise FileNotFoundError(f"Missing T1w image for {subject_id}")
    return str(t1w.resolve())

t1w_resolver = Node(
    Function(
        input_names=["subject_id", "bids_root"],
        output_names=["t1w"],
        function=get_t1w_path,
        imports=["from pathlib import Path"],
    ),
    name="t1w_resolver"
)
t1w_resolver.overwrite = True

t1w_resolver.inputs.bids_root = str(bids_root.resolve())

# -----------------------------------------------------
# Brain extraction (BET)
# -----------------------------------------------------
HAS_FSL_BET = shutil.which("bet") is not None
bet = None
if HAS_FSL_BET:
    bet = Node(
        BET(
            frac=0.5,
            robust=True,
            mask=True
        ),
        name="bet"
    )
else:
    log.warning("FSL BET not found; skipping brain extraction.")

# -----------------------------------------------------
# Tissue segmentation (FAST)
# -----------------------------------------------------
HAS_FSL_FAST = shutil.which("fast") is not None
fast = None
if HAS_FSL_FAST:
    fast = Node(
        FAST(
            segments=True,
            output_biascorrected=True
        ),
        name="fast"
    )
else:
    log.warning("FSL FAST not found; skipping tissue segmentation.")

# -----------------------------------------------------
# FreeSurfer recon-all
# -----------------------------------------------------
HAS_FREESURFER = shutil.which("recon-all") is not None
reconall = None
if HAS_FREESURFER:
    reconall = Node(
        ReconAll(
            directive="all",
            openmp=4
        ),
        name="reconall"
    )
    reconall.inputs.subjects_dir = str(freesurfer_dir)
else:
    log.warning("FreeSurfer recon-all not found; skipping cortical reconstruction.")

# -----------------------------------------------------
# STEP 6 — Subject-level volumetry
# -----------------------------------------------------
def compute_tissue_volumes(seg_files, subject_id, output_dir):
    import numpy as np
    import nibabel as nib
    import pandas as pd
    from pathlib import Path

    tissue_labels = {
        "CSF": 0,
        "GM": 1,
        "WM": 2
    }

    volumes = {"subject_id": subject_id}

    for seg_file in seg_files:
        img = nib.load(seg_file)
        data = img.get_fdata()
        voxel_volume = np.prod(img.header.get_zooms())

        for name, label in tissue_labels.items():
            volumes[name] = np.sum(data == label) * voxel_volume

    df = pd.DataFrame([volumes])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"{subject_id}_tissue_volumes.tsv"
    df.to_csv(out_file, sep="\t", index=False)

    return str(out_file)

volume_node = Node(
    Function(
        input_names=["seg_files", "subject_id", "output_dir"],
        output_names=["out_file"],
        function=compute_tissue_volumes,
        imports=[
            "import numpy as np",
            "import nibabel as nib",
            "import pandas as pd",
            "from pathlib import Path",
        ],
    ),
    name="volume_extraction"
)

volume_node.inputs.output_dir = str((pipeline_dir / "metrics").resolve())
volume_node.overwrite = True

# -----------------------------------------------------
# Connections
# -----------------------------------------------------
connections = []

# subject_id → T1 resolver
connections.append(
    (inputnode, t1w_resolver, [("subject_id", "subject_id")])
)

# BET
if bet is not None:
    connections.append(
        (t1w_resolver, bet, [("t1w", "in_file")])
    )

# FAST
if fast is not None:
    if bet is not None:
        connections.append(
            (bet, fast, [("out_file", "in_files")])
        )
    else:
        connections.append(
            (t1w_resolver, fast, [("t1w", "in_files")])
        )

    connections.append(
        (fast, volume_node, [("partial_volume_files", "seg_files")])
    )

# subject_id → volumetry
connections.append(
    (inputnode, volume_node, [("subject_id", "subject_id")])
)

# FreeSurfer
if reconall is not None:
    connections.append(
        (t1w_resolver, reconall, [("t1w", "T1_files")])
    )
    connections.append(
        (inputnode, reconall, [("subject_id", "subject_id")])
    )

wf.connect(connections)

# -----------------------------------------------------
# Execution
# -----------------------------------------------------
if __name__ == "__main__":
    log.info("Starting structural MRI pipeline")

    wf.run(
        plugin="MultiProc",
        plugin_args={
            "n_procs": 4,
            "memory_gb": 8
        }
    )
