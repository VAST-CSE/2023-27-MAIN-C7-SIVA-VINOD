import os
import glob
import nibabel as nib

image_dir = r"dataset\final\images"
label_dir = r"dataset\final\labels"

images = sorted(glob.glob(os.path.join(image_dir, "*_MRI.nii")))

print(f"Checking {len(images)} subjects...")

mismatches = []

for image_path in images:
    filename = os.path.basename(image_path)
    subject = filename.replace("_MRI.nii", "")

    label_path = os.path.join(
        label_dir,
        f"{subject}_hippocampus.nii.gz"
    )

    if not os.path.exists(label_path):
        mismatches.append(
            (subject, "LABEL_MISSING")
        )
        continue

    image = nib.load(image_path)
    label = nib.load(label_path)

    image_shape = image.shape
    label_shape = label.shape

    image_spacing = tuple(
        round(float(v), 4)
        for v in image.header.get_zooms()[:3]
    )

    label_spacing = tuple(
        round(float(v), 4)
        for v in label.header.get_zooms()[:3]
    )

    if image_shape != label_shape or image_spacing != label_spacing:
        mismatches.append(
            (
                subject,
                image_shape,
                label_shape,
                image_spacing,
                label_spacing
            )
        )

print(f"Geometry mismatches: {len(mismatches)}")

if mismatches:
    print("\nMismatches:")
    for item in mismatches:
        print(item)
else:
    print("All MRI and hippocampus labels have matching geometry.")
