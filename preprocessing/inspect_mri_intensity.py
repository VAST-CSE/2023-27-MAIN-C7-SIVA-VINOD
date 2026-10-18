import os
import glob
import numpy as np
import nibabel as nib

image_dir = r"dataset\final\images"
images = sorted(glob.glob(os.path.join(image_dir, "*_MRI.nii")))

print(f"Checking {len(images)} MRI images...")

stats = []

for image_path in images:
    subject = os.path.basename(image_path).replace("_MRI.nii", "")

    data = np.asanyarray(nib.load(image_path).dataobj).astype(np.float32)

    stats.append({
        "subject": subject,
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "nonzero": int(np.count_nonzero(data)),
        "total": int(data.size)
    })

mins = [x["min"] for x in stats]
maxs = [x["max"] for x in stats]
means = [x["mean"] for x in stats]
stds = [x["std"] for x in stats]

print("\nIntensity statistics across all subjects:")
print(f"Minimum intensity: {min(mins):.2f}")
print(f"Maximum intensity: {max(maxs):.2f}")
print(f"Mean intensity range: {min(means):.2f} - {max(means):.2f}")
print(f"STD range: {min(stds):.2f} - {max(stds):.2f}")

print("\nFirst 10 subjects:")
for x in stats[:10]:
    print(
        x["subject"],
        "min=", round(x["min"], 2),
        "max=", round(x["max"], 2),
        "mean=", round(x["mean"], 2),
        "std=", round(x["std"], 2)
    )
