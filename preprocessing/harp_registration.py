import SimpleITK as sitk
from pathlib import Path
import csv

# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = Path(
    "dataset/metadata/harp_matching_subjects.csv"
)

LABEL_DIR = Path(
    "dataset/harp_labels/Labels_100_NIFTI_Native_Space"
)

ADNI_ROOT = Path(
    "dataset/adni_downloads/ADNI"
)

OUTPUT_ROOT = Path(
    "results/registration_test"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

# Subjects already manually tested and QC checked
ALREADY_TESTED = {
    "002_S_0295",
    "002_S_0413",
    "002_S_0559",
    "002_S_0729",
    "002_S_0782",
}


# ============================================================
# FIND MRI
# ============================================================

def find_mri(subject):

    subject_dir = ADNI_ROOT / subject

    if not subject_dir.exists():
        raise FileNotFoundError(
            f"ADNI folder not found: {subject_dir}"
        )

    nii_files = list(
        subject_dir.rglob("*.nii")
    ) + list(
        subject_dir.rglob("*.nii.gz")
    )

    if not nii_files:
        raise FileNotFoundError(
            f"No NIfTI file found for {subject}"
        )

    # Prefer HarP MRI
    harp_files = [
        f for f in nii_files
        if "HarP_135_final_release_2015" in str(f)
    ]

    if harp_files:
        return harp_files[0]

    return nii_files[0]


# ============================================================
# FIND HIPPOCAMPUS LABEL
# ============================================================

def find_label(subject, side):

    pattern = f"ADNI_{subject}_*_{side}.nii"

    candidates = list(
        LABEL_DIR.glob(pattern)
    )

    # Remove CSF labels
    candidates = [
        f for f in candidates
        if "CSF" not in f.name.upper()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No hippocampus {side} label found"
        )

    return candidates[0]


# ============================================================
# BOUNDING BOX
# ============================================================

def get_bbox(image):

    arr = sitk.GetArrayFromImage(image)

    coords = (arr > 0).nonzero()

    if len(coords[0]) == 0:
        return "EMPTY"

    z_min = int(coords[0].min())
    z_max = int(coords[0].max())

    y_min = int(coords[1].min())
    y_max = int(coords[1].max())

    x_min = int(coords[2].min())
    x_max = int(coords[2].max())

    return (
        f"x={x_min}:{x_max};"
        f"y={y_min}:{y_max};"
        f"z={z_min}:{z_max}"
    )


# ============================================================
# REGISTER ONE SUBJECT
# ============================================================

def register_subject(subject):

    print("\n" + "=" * 70)
    print(f"PROCESSING: {subject}")
    print("=" * 70)

    mri_path = find_mri(subject)

    left_path = find_label(
        subject,
        "L"
    )

    right_path = find_label(
        subject,
        "R"
    )

    print("\nMRI:")
    print(mri_path)

    print("\nLEFT:")
    print(left_path)

    print("\nRIGHT:")
    print(right_path)

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    subject_output = (
        OUTPUT_ROOT / subject
    )

    subject_output.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Read MRI
    # --------------------------------------------------------

    mri = sitk.ReadImage(
        str(mri_path),
        sitk.sitkFloat32
    )

    # --------------------------------------------------------
    # Read labels
    # --------------------------------------------------------

    left = sitk.ReadImage(
        str(left_path),
        sitk.sitkUInt8
    )

    right = sitk.ReadImage(
        str(right_path),
        sitk.sitkUInt8
    )

    # --------------------------------------------------------
    # Float copy for registration
    # --------------------------------------------------------

    left_float = sitk.Cast(
        left,
        sitk.sitkFloat32
    )

    # --------------------------------------------------------
    # Initial transform
    # --------------------------------------------------------

    initial_transform = (
        sitk.CenteredTransformInitializer(
            mri,
            left_float,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    )

    # --------------------------------------------------------
    # Registration
    # --------------------------------------------------------

    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50
    )

    registration.SetMetricSamplingStrategy(
        registration.RANDOM
    )

    registration.SetMetricSamplingPercentage(
        0.02
    )

    registration.SetInterpolator(
        sitk.sitkLinear
    )

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.001,
        numberOfIterations=100,
        gradientMagnitudeTolerance=1e-6
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetInitialTransform(
        initial_transform,
        inPlace=False
    )

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    final_transform = registration.Execute(
        mri,
        left_float
    )

    # --------------------------------------------------------
    # Save transform
    # --------------------------------------------------------

    transform_path = (
        subject_output /
        f"{subject}_rigid_transform.tfm"
    )

    sitk.WriteTransform(
        final_transform,
        str(transform_path)
    )

    # --------------------------------------------------------
    # Resample LEFT
    # --------------------------------------------------------

    left_registered = sitk.Resample(
        left,
        mri,
        final_transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )

    left_output = (
        subject_output /
        f"{subject}_left_registered.nii.gz"
    )

    sitk.WriteImage(
        left_registered,
        str(left_output)
    )

    # --------------------------------------------------------
    # Resample RIGHT
    # --------------------------------------------------------

    right_registered = sitk.Resample(
        right,
        mri,
        final_transform,
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )

    right_output = (
        subject_output /
        f"{subject}_right_registered.nii.gz"
    )

    sitk.WriteImage(
        right_registered,
        str(right_output)
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = sitk.Maximum(
        left_registered,
        right_registered
    )

    combined_output = (
        subject_output /
        f"{subject}_combined_hippocampus_registered.nii.gz"
    )

    sitk.WriteImage(
        combined,
        str(combined_output)
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    left_array = sitk.GetArrayViewFromImage(
        left_registered
    )

    right_array = sitk.GetArrayViewFromImage(
        right_registered
    )

    combined_array = sitk.GetArrayViewFromImage(
        combined
    )

    left_voxels = int(
        (left_array > 0).sum()
    )

    right_voxels = int(
        (right_array > 0).sum()
    )

    combined_voxels = int(
        (combined_array > 0).sum()
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if left_voxels == 0:
        raise RuntimeError(
            "Left registered mask is empty"
        )

    if right_voxels == 0:
        raise RuntimeError(
            "Right registered mask is empty"
        )

    if combined_voxels == 0:
        raise RuntimeError(
            "Combined registered mask is empty"
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    metric = registration.GetMetricValue()

    iterations = (
        registration.GetOptimizerIteration()
    )

    print("\nRESULT")
    print("-" * 50)

    print(
        "Left voxels    :",
        left_voxels
    )

    print(
        "Right voxels   :",
        right_voxels
    )

    print(
        "Combined voxels:",
        combined_voxels
    )

    print(
        "Metric         :",
        metric
    )

    print(
        "Iterations     :",
        iterations
    )

    print(
        "STATUS         : SUCCESS"
    )

    return {
        "subject": subject,
        "status": "SUCCESS",
        "metric": metric,
        "iterations": iterations,
        "left_voxels": left_voxels,
        "right_voxels": right_voxels,
        "combined_voxels": combined_voxels,
        "left_bbox": get_bbox(left_registered),
        "right_bbox": get_bbox(right_registered),
        "combined_bbox": get_bbox(combined),
        "transform": str(transform_path),
        "left_output": str(left_output),
        "right_output": str(right_output),
        "combined_output": str(combined_output),
        "error": "",
    }


# ============================================================
# READ SUBJECTS
# ============================================================

with open(
    CSV_PATH,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)


print("\n==============================================")
print("HARP AUTOMATIC REGISTRATION")
print("==============================================")

print(
    "Subjects in CSV:",
    len(rows)
)

# Find subject column automatically
subject_column = None

for possible in [
    "subject",
    "Subject",
    "SUBJECT",
    "Subject ID",
    "SubjectID",
    "PTID",
]:

    if possible in reader.fieldnames:
        subject_column = possible
        break

if subject_column is None:

    raise RuntimeError(
        "Could not find subject column in CSV. "
        f"Columns found: {reader.fieldnames}"
    )

print(
    "Subject column:",
    subject_column
)


subjects = []

for row in rows:

    subject = str(
        row[subject_column]
    ).strip()

    if subject:
        subjects.append(subject)


# Remove duplicates while preserving order
subjects = list(
    dict.fromkeys(subjects)
)

print(
    "Unique subjects:",
    len(subjects)
)

# ------------------------------------------------------------
# Determine remaining subjects
# ------------------------------------------------------------

remaining = [
    s for s in subjects
    if s not in ALREADY_TESTED
]

print(
    "Already tested:",
    len(ALREADY_TESTED)
)

print(
    "Remaining:",
    len(remaining)
)

print("\nThe following subjects will be processed:")

for s in remaining:
    print(" ", s)


# ============================================================
# RUN
# ============================================================

results = []

for index, subject in enumerate(
    remaining,
    start=1
):

    print("\n")
    print(
        f"[{index}/{len(remaining)}] {subject}"
    )

    try:

        result = register_subject(
            subject
        )

        results.append(
            result
        )

    except Exception as e:

        print("\nSTATUS: FAILED")
        print(
            "ERROR:",
            str(e)
        )

        results.append(
            {
                "subject": subject,
                "status": "FAILED",
                "metric": "",
                "iterations": "",
                "left_voxels": "",
                "right_voxels": "",
                "combined_voxels": "",
                "left_bbox": "",
                "right_bbox": "",
                "combined_bbox": "",
                "transform": "",
                "left_output": "",
                "right_output": "",
                "combined_output": "",
                "error": str(e),
            }
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = (
    OUTPUT_ROOT /
    "remaining_95_registration_summary.csv"
)

fieldnames = [
    "subject",
    "status",
    "metric",
    "iterations",
    "left_voxels",
    "right_voxels",
    "combined_voxels",
    "left_bbox",
    "right_bbox",
    "combined_bbox",
    "transform",
    "left_output",
    "right_output",
    "combined_output",
    "error",
]

with open(
    summary_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for result in results:
        writer.writerow(result)


# ============================================================
# FINAL SUMMARY
# ============================================================

successful = sum(
    1
    for r in results
    if r["status"] == "SUCCESS"
)

failed = sum(
    1
    for r in results
    if r["status"] == "FAILED"
)

print("\n")
print("=" * 70)
print("REGISTRATION COMPLETE")
print("=" * 70)

print(
    "Processed:",
    len(results)
)

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)

print(
    "Summary:",
    summary_path
)

print("\nFinished.")
