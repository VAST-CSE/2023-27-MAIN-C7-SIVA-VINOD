from pathlib import Path
import nibabel as nib

root = Path("dataset")

for subject in ["005_S_0448", "005_S_0602"]:
    print(f"\n===== {subject} =====")

    for path in root.rglob("*.nii"):
        if subject in path.name and (
            path.name.endswith("_L.nii")
            or path.name.endswith("_R.nii")
            or path.name.endswith("_CSF_L.nii")
            or path.name.endswith("_CSF_R.nii")
        ):
            img = nib.load(str(path))
            data = img.get_fdata()

            print(f"\nFile: {path.name}")
            print("Shape:", data.shape)
            print("Unique values:", sorted(set(data.ravel()))[:20])
            print("Affine:")
            print(img.affine)
