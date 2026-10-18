import os
import glob
import numpy as np
import nibabel as nib

image_dir = r"dataset\final\normalized_images"
label_dir = r"dataset\final\labels"

images = sorted(glob.glob(os.path.join(image_dir, "*_MRI_normalized.nii.gz")))

print(f"Normalized MRI files: {len(images)}")

problems = []

for image_path in images:
    subject = os.path.basename(image_path).replace(
        "_MRI_normalized.nii.gz", ""
    )

    label_path = os.path.join(
        label_dir,
        f"{subject}_hippocampus.nii.gz"
    )

    img = nib.load(image_path)
    data = np.asanyarray(img.dataobj).astype(np.float32)

    if label_path and os.path.exists(label_path):
        label = nib.load(label_path)

        if img.shape != label.shape:
            problems.append(
                f"{subject}: shape mismatch "
                f"{img.shape} vs {label.shape}"
            )

        if not np.allclose(img.header.get_zooms()[:3],
                           label.header.get_zooms()[:3]):
            problems.append(
                f"{subject}: spacing mismatch"
            )
    else:
        problems.append(f"{subject}: missing label")

    if np.isnan(data).any():
        problems.append(f"{subject}: NaN values")

    if np.isinf(data).any():
        problems.append(f"{subject}: infinite values")

    nonzero = data[data != 0]

    print(
        f"{subject}: "
        f"shape={data.shape}, "
        f"min={data.min():.3f}, "
        f"max={data.max():.3f}, "
        f"nonzero_mean={nonzero.mean():.3f}, "
        f"nonzero_std={nonzero.std():.3f}"
    )

print("\n========================================")

if problems:
    print(f"Problems found: {len(problems)}")
    for problem in problems:
        print(problem)
else:
    print("Problems found: 0")
    print("All normalized MRI images are valid.")
    print("MRI-label geometry is consistent.")
