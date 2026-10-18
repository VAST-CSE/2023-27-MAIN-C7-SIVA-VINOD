import os
import glob
import numpy as np
import nibabel as nib

input_dir = r"dataset\final\images"
output_dir = r"dataset\final\normalized_images"

os.makedirs(output_dir, exist_ok=True)

images = sorted(glob.glob(os.path.join(input_dir, "*_MRI.nii")))

print(f"Normalizing {len(images)} MRI images...")

for image_path in images:
    subject = os.path.basename(image_path).replace("_MRI.nii", "")

    img = nib.load(image_path)
    data = np.asanyarray(img.dataobj).astype(np.float32)

    nonzero = data[data > 0]

    if nonzero.size == 0:
        print(f"ERROR: {subject} has no nonzero voxels")
        continue

    mean = np.mean(nonzero)
    std = np.std(nonzero)

    if std == 0:
        print(f"ERROR: {subject} has zero standard deviation")
        continue

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[data > 0] = (data[data > 0] - mean) / std

    output_path = os.path.join(
        output_dir,
        f"{subject}_MRI_normalized.nii.gz"
    )

    out_img = nib.Nifti1Image(
        normalized,
        img.affine,
        img.header
    )

    out_img.set_data_dtype(np.float32)

    nib.save(out_img, output_path)

    print(
        f"{subject}: "
        f"mean={mean:.2f}, "
        f"std={std:.2f}, "
        f"saved"
    )

print("\nNormalization complete.")
