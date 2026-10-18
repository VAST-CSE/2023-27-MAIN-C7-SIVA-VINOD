import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

image_dir = r"dataset\final\normalized_images"
label_dir = r"dataset\final\labels"
output_dir = r"results\label_qc_all"

os.makedirs(output_dir, exist_ok=True)

labels = sorted(
    glob.glob(os.path.join(label_dir, "*_hippocampus.nii.gz"))
)

print(f"Generating QC images for {len(labels)} subjects...")
print()

for i, label_path in enumerate(labels, 1):

    subject = os.path.basename(label_path).replace(
        "_hippocampus.nii.gz", ""
    )

    image_path = os.path.join(
        image_dir,
        f"{subject}_MRI_normalized.nii.gz"
    )

    if not os.path.exists(image_path):
        print(f"Missing MRI: {subject}")
        continue

    image = np.asanyarray(
        nib.load(image_path).dataobj
    )

    label = np.asanyarray(
        nib.load(label_path).dataobj
    )

    coords = np.argwhere(label > 0)

    if coords.size == 0:
        print(f"Empty label: {subject}")
        continue

    center = coords.mean(axis=0).astype(int)

    x, y, z = center

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial
    axes[0].imshow(
        image[:, :, z].T,
        cmap="gray",
        origin="lower"
    )

    axes[0].imshow(
        np.ma.masked_where(
            label[:, :, z].T == 0,
            label[:, :, z].T
        ),
        cmap="Reds",
        alpha=0.5,
        origin="lower"
    )

    axes[0].set_title(f"Axial z={z}")

    # Coronal
    axes[1].imshow(
        image[:, y, :].T,
        cmap="gray",
        origin="lower"
    )

    axes[1].imshow(
        np.ma.masked_where(
            label[:, y, :].T == 0,
            label[:, y, :].T
        ),
        cmap="Reds",
        alpha=0.5,
        origin="lower"
    )

    axes[1].set_title(f"Coronal y={y}")

    # Sagittal
    axes[2].imshow(
        image[x, :, :].T,
        cmap="gray",
        origin="lower"
    )

    axes[2].imshow(
        np.ma.masked_where(
            label[x, :, :].T == 0,
            label[x, :, :].T
        ),
        cmap="Reds",
        alpha=0.5,
        origin="lower"
    )

    axes[2].set_title(f"Sagittal x={x}")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(
        f"{subject} - Hippocampus Label QC"
    )

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        f"{subject}_qc.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    if i % 10 == 0:
        print(f"Generated {i}/{len(labels)}")

print()
print("====================================")
print("QC image generation complete.")
print(f"Output folder: {output_dir}")
print("====================================")
