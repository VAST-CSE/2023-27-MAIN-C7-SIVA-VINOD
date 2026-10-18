import os
import pandas as pd
import numpy as np
import torch

from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandCropByPosNegLabeld
)

split_csv = r"dataset\final\metadata\train_val_test_split.csv"
image_dir = r"dataset\final\normalized_images"
label_dir = r"dataset\final\labels"

df = pd.read_csv(split_csv)

train_df = df[df["Split"] == "train"]

items = []

for subject in train_df["Subject ID"]:
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

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1,
        neg=0,
        num_samples=1
    )
])

dataset = Dataset(
    data=items,
    transform=transforms
)

print(f"Training subjects: {len(items)}")
print()
print("Sampling 20 positive patches...")
print()

fractions = []

for i in range(20):

    samples = dataset[i % len(dataset)]
    sample = samples[0]

    label = sample["label"]

    foreground = torch.sum(label > 0).item()
    total = label.numel()

    fraction = foreground / total

    fractions.append(fraction)

    print(
        f"Patch {i+1:02d}: "
        f"foreground voxels = {foreground:6d} / {total} "
        f"({fraction*100:.3f}%)"
    )

print()
print("======================================")
print("Patch foreground statistics")
print("======================================")

print(f"Minimum: {min(fractions)*100:.3f}%")
print(f"Maximum: {max(fractions)*100:.3f}%")
print(f"Mean:    {np.mean(fractions)*100:.3f}%")
print(f"Median:  {np.median(fractions)*100:.3f}%")

print()
print("Patch balance diagnostic complete.")
