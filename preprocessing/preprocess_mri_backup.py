
from pathlib import Path

import nibabel as nib
import numpy as np

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = DATASET_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find converted MRI volumes, excluding previous outputs
mri_files = [
    p for p in DATASET_DIR.rglob("*")
    if p.is_file()
    and (p.name.lower().endswith(".nii")
         or p.name.lower().endswith(".nii.gz"))
    and OUTPUT_DIR not in p.parents
]

if not mri_files:
    print("No NIfTI MRI files found in dataset.")
    raise SystemExit(1)

print(f"Found {len(mri_files)} MRI files.")

for mri_path in mri_files:
    try:
        image = nib.load(str(mri_path))
        volume = image.get_fdata(dtype=np.float32)

        finite = np.isfinite(volume)
        if not finite.all():
            print(f"Skipping invalid values: {mri_path.name}")
            continue

        nonzero = volume != 0
        if not nonzero.any():
            print(f"Skipping empty image: {mri_path.name}")
            continue

        # Normalize non-background voxels using robust percentiles.
        values = volume[nonzero]
        low, high = np.percentile(values, [1, 99])

        if high <= low:
            print(f"Skipping constant image: {mri_path.name}")
            continue

        processed = np.zeros_like(volume)
        processed[nonzero] = np.clip(
            (volume[nonzero] - low) / (high - low),
            0.0,
            1.0,
        )

        # Preserve the original spatial metadata.
        output_image = nib.Nifti1Image(
            processed, image.affine, image.header.copy()
        )
        output_image.set_data_dtype(np.float32)

        output_path = OUTPUT_DIR / f"{mri_path.name}"
        nib.save(output_image, str(output_path))

        print(f"Saved: {output_path.name}")
        print(f"  Dimensions: {volume.shape}")
        print(f"  Voxel spacing: {image.header.get_zooms()[:3]}")

    except Exception as exc:
        print(f"Error processing {mri_path.name}: {exc}")

print("MRI preprocessing finished.")

INPUT_DIR = PROJECT_ROOT / "dataset" / "nifti"

mri_files = sorted(
    list(INPUT_DIR.glob("*.nii")) +
    list(INPUT_DIR.glob("*.nii.gz"))
)