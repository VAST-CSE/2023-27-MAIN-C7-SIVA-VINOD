from pathlib import Path
import nibabel as nib

root = Path("dataset")

for subject in ["005_S_0448", "005_S_0602"]:
    print(f"\n===== {subject} =====")

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if subject not in path.name:
            continue

        if path.suffix not in [".nii", ".gz"]:
            continue

        if "_MR_HarP_135_final_release_2015_" in path.name:
            continue

        if path.name.endswith(("_L.nii", "_R.nii", "_CSF_L.nii", "_CSF_R.nii")):
            img = nib.load(str(path))
            print("\nFile:", path.name)
            print("Shape:", img.shape)
            print("Orientation:", nib.aff2axcodes(img.affine))
            print("Affine:")
            print(img.affine)
