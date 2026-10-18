from pathlib import Path
import nibabel as nib

root = Path("dataset")

for subject in ["005_S_0448", "005_S_0602"]:
    print(f"\n===== {subject} =====")

    files = list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz"))

    for path in files:
        if subject not in path.name:
            continue

        try:
            img = nib.load(str(path))
            print(f"\nFile: {path.name}")
            print(f"Shape: {img.shape}")
            print(f"Voxel spacing: {img.header.get_zooms()[:3]}")
            print(f"Orientation: {nib.aff2axcodes(img.affine)}")
        except Exception as e:
            print(f"Could not read {path.name}: {e}")
