import os
import glob
import random
import csv

image_dir = r"dataset\final\normalized_images"
label_dir = r"dataset\final\labels"
output_dir = r"dataset\final\metadata"

os.makedirs(output_dir, exist_ok=True)

images = sorted(glob.glob(os.path.join(
    image_dir, "*_MRI_normalized.nii.gz"
)))

subjects = []

for image_path in images:
    subject = os.path.basename(image_path).replace(
        "_MRI_normalized.nii.gz", ""
    )

    label_path = os.path.join(
        label_dir,
        f"{subject}_hippocampus.nii.gz"
    )

    if os.path.exists(label_path):
        subjects.append(subject)

print(f"Valid subjects: {len(subjects)}")

if len(subjects) != 100:
    raise RuntimeError(
        f"Expected 100 valid subjects, found {len(subjects)}"
    )

random.seed(42)
random.shuffle(subjects)

train = subjects[:70]
validation = subjects[70:85]
test = subjects[85:100]

splits = []

for subject in train:
    splits.append([subject, "train"])

for subject in validation:
    splits.append([subject, "validation"])

for subject in test:
    splits.append([subject, "test"])

output_csv = os.path.join(
    output_dir,
    "train_val_test_split.csv"
)

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Subject ID", "Split"])
    writer.writerows(splits)

print("\nDataset split:")
print(f"Training:   {len(train)}")
print(f"Validation: {len(validation)}")
print(f"Testing:    {len(test)}")

print(f"\nSaved to:")
print(output_csv)
