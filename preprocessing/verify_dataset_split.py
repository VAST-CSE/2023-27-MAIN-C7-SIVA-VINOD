import csv

split_file = r"dataset\final\metadata\train_val_test_split.csv"

splits = {
    "train": set(),
    "validation": set(),
    "test": set()
}

with open(split_file, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        subject = row["Subject ID"]
        split = row["Split"]
        splits[split].add(subject)

train_val = splits["train"] & splits["validation"]
train_test = splits["train"] & splits["test"]
val_test = splits["validation"] & splits["test"]

print("Train subjects:", len(splits["train"]))
print("Validation subjects:", len(splits["validation"]))
print("Test subjects:", len(splits["test"]))

print("\nOverlap checks:")
print("Train n Validation:", len(train_val))
print("Train n Test:", len(train_test))
print("Validation n Test:", len(val_test))

if train_val or train_test or val_test:
    print("\nERROR: Subject overlap detected!")
else:
    print("\nSUCCESS: No subject overlap.")
    print("Dataset split is ready for model development.")
