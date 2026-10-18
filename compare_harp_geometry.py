from pathlib import Path
import nibabel as nib
import numpy as np

root = Path("dataset")

for subject in ["005_S_0448", "005_S_0602"]:
    print(f"\n===== {subject} =====")

    images = [
        p for p in root.rglob("*.nii")
        if subject in p.name and "_MR_HarP_135_final_release_2015_" in p.name
    ]

    labels = [
        p for p in root.rglob("*.nii")
        if subject in p.name and p.name.endswith(("_L.nii", "_R.nii"))
        and "_MR_HarP_135_final_release_2015_" not in p.name
    ]

    for path in images + labels:
        img = nib.load(str(path))
        data = np.asarray(img.dataobj)

        print(f"\n{path.name}")
        print("Shape:", img.shape)
        print("Orientation:", nib.aff2axcodes(img.affine))
        print("Affine:")
        print(img.affine)
