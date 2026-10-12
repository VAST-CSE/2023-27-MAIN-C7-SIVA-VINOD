
from pathlib import Path
import nibabel as nib
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NIFTI_DIR = PROJECT_ROOT / "dataset" / "nifti"

files = list(NIFTI_DIR.glob("*.nii.gz"))

if not files:
    print(f"No NIfTI files found in: {NIFTI_DIR}")
    raise SystemExit(1)

print(f"Found {len(files)} NIfTI files\n")

for file_path in sorted(files):
    try:
        img = nib.load(str(file_path))
        data = img.get_fdata(dtype=np.float32)

        print(f"File: {file_path.name}")
        print(f"  Dimensions: {img.shape}")
        print(f"  Voxel spacing (mm): {img.header.get_zooms()}")
        print(f"  Orientation: {nib.aff2axcodes(img.affine)}")
        print(f"  Intensity range: {np.nanmin(data):.3f} to {np.nanmax(data):.3f}")
        print(f"  Contains non-finite values: {not np.isfinite(data).all()}")
        print(f"  File readable: Yes\n")

    except Exception as exc:
        print(f"ERROR: {file_path.name}: {exc}\n")
