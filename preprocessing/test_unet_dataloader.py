import os
import csv
import torch

from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandCropByPosNegLabeld
)

split_file = r"dataset\final\metadata\train_val_test_split.csv"
image_dir = r"dataset\final\normalized_images"
label_dir = r"dataset\final\labels"

items = []

with open(split_file, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["Split"] == "train":
            subject = row["Subject ID"]

            items.append({
                "image": os.path.join(
                    image_dir,
                    f"{subject}_MRI_normalized.nii.gz"
                ),
                "label": os.path.join(
                    label_dir,
                    f"{subject}_hippocampus.nii.gz"
                )
            })

print(f"Training subjects found: {len(items)}")

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1,
        neg=1,
        num_samples=1,
        image_key="image",
        image_threshold=0
    )
])

dataset = CacheDataset(
    data=items[:2],
    transform=transforms,
    cache_rate=1.0
)

samples = dataset[0]

print("\nNumber of patches returned:", len(samples))

sample = samples[0]

print("Sample loaded successfully.")
print("Image shape:", sample["image"].shape)
print("Label shape:", sample["label"].shape)
print("Image dtype:", sample["image"].dtype)
print("Label dtype:", sample["label"].dtype)
print("Image min:", float(sample["image"].min()))
print("Image max:", float(sample["image"].max()))
print("Label unique values:", torch.unique(sample["label"]))

print("\n3D patch loading test successful.")
