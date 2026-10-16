
from pathlib import Path
import hashlib

import nibabel as nib
import numpy as np


# ==========================================
# 1. PROJECT DIRECTORIES
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "dataset" / "nifti"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "preprocessed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. FIND INPUT MRI FILES
# ==========================================

mri_files = sorted(
    list(INPUT_DIR.glob("*.nii")) +
    list(INPUT_DIR.glob("*.nii.gz"))
)

if not mri_files:
    raise SystemExit(
        f"No NIfTI MRI files found in: {INPUT_DIR}"
    )

print(f"Found {len(mri_files)} MRI files.")


# ==========================================
# 3. REMOVE DUPLICATE SCANS
#    (BY CONTENT, NOT FILENAME)
# ==========================================

unique_files = []
seen_hashes = set()

for mri_path in mri_files:
    try:
        digest = hashlib.sha256(mri_path.read_bytes()).hexdigest()

        if digest in seen_hashes:
            print(f"Duplicate scan skipped: {mri_path.name}")
            continue

        seen_hashes.add(digest)
        unique_files.append(mri_path)

    except OSError as exc:
        print(f"Cannot read {mri_path.name}: {exc}")

print(f"Unique MRI scans: {len(unique_files)}")


# ==========================================
# 4. PREPROCESS MRI VOLUMES
# ==========================================

successful = 0
skipped = 0
failed = 0

for mri_path in unique_files:
    try:
        print(f"\nProcessing: {mri_path.name}")

        # Load MRI volume
        image = nib.load(str(mri_path))
        volume = image.get_fdata(dtype=np.float32)

        # Check for invalid values
        if not np.isfinite(volume).all():
            print("Skipped: MRI contains NaN or infinite values.")
            skipped += 1
            continue

        # Check for empty image
        nonzero = volume != 0

        if not nonzero.any():
            print("Skipped: MRI volume is empty.")
            skipped += 1
            continue

        # Robust intensity normalization using the
        # 1st and 99th percentiles of non-background voxels
        values = volume[nonzero]
        low, high = np.percentile(values, [1, 99])

        if high <= low:
            print("Skipped: MRI has insufficient intensity variation.")
            skipped += 1
            continue

        processed = np.zeros_like(volume, dtype=np.float32)

        processed[nonzero] = np.clip(
            (volume[nonzero] - low) / (high - low),
            0.0,
            1.0,
        )

        # Preserve spatial affine and image header
        output_image = nib.Nifti1Image(
            processed,
            image.affine,
            image.header.copy(),
        )

        output_image.set_data_dtype(np.float32)

        # Save normalized MRI
        output_path = OUTPUT_DIR / mri_path.name
        nib.save(output_image, str(output_path))

        # Report basic quality information
        print(f"Saved: {output_path}")
        print(f"Dimensions: {volume.shape}")
        print(f"Voxel spacing: {image.header.get_zooms()[:3]}")
        print(
            f"Normalized intensity range: "
            f"{processed.min():.3f} to {processed.max():.3f}"
        )

        successful += 1

    except Exception as exc:
        print(f"Error processing {mri_path.name}: {exc}")
        failed += 1


# ==========================================
# 5. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 45)
print("MRI PREPROCESSING SUMMARY")
print("=" * 45)
print(f"Input files found: {len(mri_files)}")
print(f"Unique scans identified: {len(unique_files)}")
print(f"Successfully processed: {successful}")
print(f"Skipped: {skipped}")
print(f"Failed: {failed}")
print(f"Output directory: {OUTPUT_DIR}")
print("MRI intensity normalization finished.")
