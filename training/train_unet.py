import os
import csv
import torch

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandCropByPosNegLabeld
)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

# --------------------------------------------------
# Paths
# --------------------------------------------------

split_file = r"dataset\final\metadata\train_val_test_split.csv"
image_dir = r"dataset\final\normalized_images"
label_dir = r"dataset\final\labels"
checkpoint_dir = r"results\training"

os.makedirs(checkpoint_dir, exist_ok=True)

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# --------------------------------------------------
# Read dataset split
# --------------------------------------------------

train_items = []
val_items = []

with open(split_file, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        subject = row["Subject ID"]

        item = {
            "image": os.path.join(
                image_dir,
                f"{subject}_MRI_normalized.nii.gz"
            ),
            "label": os.path.join(
                label_dir,
                f"{subject}_hippocampus.nii.gz"
            )
        }

        if row["Split"] == "train":
            train_items.append(item)

        elif row["Split"] == "validation":
            val_items.append(item)

print("Training subjects:", len(train_items))
print("Validation subjects:", len(val_items))

# --------------------------------------------------
# Training transforms
# --------------------------------------------------

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=4,
        neg=1,
        num_samples=1,
        image_key="image",
        image_threshold=0
    )
])

# --------------------------------------------------
# Validation transforms
# --------------------------------------------------

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"])
])

# --------------------------------------------------
# Datasets
# --------------------------------------------------

train_ds = CacheDataset(
    data=train_items,
    transform=train_transforms,
    cache_rate=0.2,
    num_workers=0
)

val_ds = CacheDataset(
    data=val_items,
    transform=val_transforms,
    cache_rate=0.2,
    num_workers=0
)

# --------------------------------------------------
# Data loaders
# --------------------------------------------------

train_loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# 3D U-Net
# --------------------------------------------------

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2
).to(device)

print("\n3D U-Net created.")

# --------------------------------------------------
# Loss and optimizer
# --------------------------------------------------

loss_function = DiceCELoss(
    to_onehot_y=True,
    softmax=True
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# --------------------------------------------------
# Smoke-test training
# --------------------------------------------------

epochs = 10

print("\nStarting smoke-test training...")
print("Epochs:", epochs)

for epoch in range(epochs):

    model.train()
    epoch_loss = 0.0

    for step, batch in enumerate(train_loader):

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Step {step + 1}/{len(train_loader)} "
            f"Loss: {loss.item():.4f}"
        )

    epoch_loss /= len(train_loader)

    print(
        f"\nEpoch {epoch + 1} completed. "
        f"Average loss: {epoch_loss:.4f}\n"
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"unet_epoch_{epoch + 1}.pth"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss
        },
        checkpoint_path
    )

    print("Checkpoint saved:", checkpoint_path)

print("\nSmoke-test training completed successfully.")
