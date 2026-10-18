import os
import glob
import numpy as np
import nibabel as nib

label_dir = r"dataset\final\labels"

labels = sorted(
    glob.glob(os.path.join(
        label_dir,
        "*_hippocampus.nii.gz"
    ))
)

print(f"Checking {len(labels)} labels...\n")

records = []

for path in labels:

    subject = os.path.basename(path).replace(
        "_hippocampus.nii.gz", ""
    )

    data = np.asanyarray(
        nib.load(path).dataobj
    )

    coords = np.argwhere(data > 0)

    if coords.size == 0:
        print(f"{subject}: EMPTY")
        continue

    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)

    center = (min_xyz + max_xyz) / 2
    volume = len(coords)

    records.append({
        "subject": subject,
        "volume": volume,
        "min_x": min_xyz[0],
        "max_x": max_xyz[0],
        "min_y": min_xyz[1],
        "max_y": max_xyz[1],
        "min_z": min_xyz[2],
        "max_z": max_xyz[2],
        "cx": center[0],
        "cy": center[1],
        "cz": center[2]
    })

print("\nLabel bounding-box summary")
print("==========================")

for r in records:
    print(
        f"{r['subject']}: "
        f"voxels={r['volume']}, "
        f"bbox=({r['min_x']}:{r['max_x']}, "
        f"{r['min_y']}:{r['max_y']}, "
        f"{r['min_z']}:{r['max_z']}), "
        f"center=({r['cx']:.1f}, "
        f"{r['cy']:.1f}, "
        f"{r['cz']:.1f})"
    )

print("\n========================================")

centers = np.array([
    [r["cx"], r["cy"], r["cz"]]
    for r in records
])

volumes = np.array([
    r["volume"]
    for r in records
])

print(
    "Center range:"
    f"\nX: {centers[:,0].min():.1f} - {centers[:,0].max():.1f}"
    f"\nY: {centers[:,1].min():.1f} - {centers[:,1].max():.1f}"
    f"\nZ: {centers[:,2].min():.1f} - {centers[:,2].max():.1f}"
)

print(
    "\nVolume range:"
    f"\nMinimum: {volumes.min()}"
    f"\nMaximum: {volumes.max()}"
    f"\nMean: {volumes.mean():.2f}"
    f"\nMedian: {np.median(volumes):.2f}"
)

print("\nLocation/volume sanity check complete.")
