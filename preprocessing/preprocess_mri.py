from pathlib import Path
import hashlib

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "dataset" / "nifti"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "preprocessed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Read ONLY the converted MRI files
mri_files = sorted(
    list(INPUT_DIR.glob("*.nii")) +
    list(INPUT_DIR.glob("*.nii.gz"))
)

if not mri_files:
    raise SystemExit(f"No NIfTI files found in {INPUT_DIR}")

print(f"Input MRI files found: {len(mri_files)}")

# Skip files with identical contents
unique_files = []
seen_hashes = set()

for path in mri_files:
    try:
        digest = sha256_file(path)
    except OSError as exc:
        print(f"Cannot read {path.name}: {exc}")
        continue

    if digest in seen_hashes:
        print(f"Duplicate scan skipped: {path.name}")
        continue

    seen_hashes.add(digest)
    unique_files.append(path)

print(f"Unique MRI scans: {len(unique_files)}")

successful = 0
skipped = 0
failed = 0

for mri_path in unique_files:
    try:
        print(f"\nProcessing: {mri_path.name}")

        image = nib.load(str(mri_path))
        volume = image.get_fdata(dtype=np.float32)

        if not np.isfinite(volume).all():
            print("Skipped: contains NaN or infinite values.")
            skipped += 1
            continue

        nonzero = volume != 0

        if not nonzero.any():
            print("Skipped: empty MRI volume.")
            skipped += 1
            continue

        values = volume[nonzero]
        low, high = np.percentile(values, [1, 99])

        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            print("Skipped: invalid intensity range.")
            skipped += 1
            continue

        processed = np.zeros_like(volume, dtype=np.float32)
        processed[nonzero] = np.clip(
            (volume[nonzero] - low) / (high - low),
            0.0,
            1.0,
        )

        output_image = nib.Nifti1Image(
            processed,
            image.affine,
            image.header.copy(),
        )
        output_image.set_data_dtype(np.float32)

        output_path = OUTPUT_DIR / mri_path.name
        nib.save(output_image, str(output_path))

        print(f"Saved: {output_path.name}")
        print(f"Dimensions: {volume.shape}")
        print(f"Voxel spacing: {image.header.get_zooms()[:3]}")
        print(
            f"Normalized range: {processed.min():.3f} "
            f"to {processed.max():.3f}"
        )

        successful += 1

    except Exception as exc:
        print(f"Error processing {mri_path.name}: {exc}")
        failed += 1

print("\n" + "=" * 40)
print("MRI PREPROCESSING SUMMARY")
print("=" * 40)
print(f"Input files: {len(mri_files)}")
print(f"Unique scans: {len(unique_files)}")
print(f"Successfully processed: {successful}")
print(f"Skipped: {skipped}")
print(f"Failed: {failed}")
print(f"Output directory: {OUTPUT_DIR}")
