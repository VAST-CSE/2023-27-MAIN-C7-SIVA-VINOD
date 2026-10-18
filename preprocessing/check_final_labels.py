import os
import glob
import numpy as np
import nibabel as nib

label_dir = r"dataset\final\labels"
labels = sorted(glob.glob(os.path.join(label_dir, "*_hippocampus.nii.gz")))

print(f"Checking {len(labels)} labels...")

problems = []
voxel_counts = []

for label_path in labels:
    subject = os.path.basename(label_path).replace(
        "_hippocampus.nii.gz", ""
    )

    data = np.asanyarray(nib.load(label_path).dataobj)

    unique_values = np.unique(data)
    foreground = int(np.count_nonzero(data))

    voxel_counts.append(foreground)

    if not np.all(np.isin(unique_values, [0, 1])):
        problems.append(
            (subject, "Unexpected label values", unique_values.tolist())
        )

    if foreground == 0:
        problems.append(
            (subject, "Empty label", foreground)
        )

print(f"Labels with problems: {len(problems)}")

if problems:
    print("\nProblems:")
    for item in problems:
        print(item)

print("\nForeground voxel statistics:")
print(f"Minimum: {min(voxel_counts)}")
print(f"Maximum: {max(voxel_counts)}")
print(f"Mean:    {np.mean(voxel_counts):.2f}")
print(f"Median:  {np.median(voxel_counts):.2f}")

if not problems:
    print("\nAll labels are non-empty binary masks.")
