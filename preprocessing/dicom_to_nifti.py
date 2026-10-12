
from pathlib import Path
import csv
import re
import logging

import pydicom
import SimpleITK as sitk
import nibabel as nib
import numpy as np


# Project root: parent of preprocessing/
ROOT = Path(__file__).resolve().parent.parent

DICOM_ROOT = Path(r"D:\ADNI_T1_MRI_Baseline\ADNI")
OUTPUT_DIR = ROOT / "dataset" / "nifti"
REPORT_PATH = ROOT / "results" / "dicom_conversion_report.csv"

OVERWRITE_EXISTING = False

REPORT_FIELDS = [
    "subject_id",
    "series_instance_uid",
    "series_description",
    "status",
    "number_of_slices",
    "dimensions",
    "voxel_spacing_mm",
    "output_filename",
    "errors",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def subject_from_path(path: Path) -> str:
    """Find an ADNI subject ID such as 005_S_0448 in the path."""
    for part in path.parts:
        match = re.search(r"\b\d{3}_S_\d{4}\b", part, re.I)
        if match:
            return match.group(0).upper()
    return ""


def discover_series(root: Path):
    """Read DICOM headers and group files by SeriesInstanceUID."""
    groups = {}
    errors = []

    if not root.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {root}")

    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".dcm", ".dicom", ""}
    ]

    # Also inspect files with unusual extensions by checking their headers.
    logger.info("Inspecting %d candidate files", len(files))

    for path in files:
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
            )

            uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
            if not uid:
                continue

            # Avoid accidentally grouping unrelated non-image objects.
            if not hasattr(ds, "SOPInstanceUID"):
                continue

            position = getattr(ds, "ImagePositionPatient", None)
            orientation = getattr(ds, "ImageOrientationPatient", None)
            instance = getattr(ds, "InstanceNumber", None)

            record = {
                "path": path,
                "position": (
                    [float(v) for v in position]
                    if position is not None else None
                ),
                "orientation": (
                    [float(v) for v in orientation]
                    if orientation is not None else None
                ),
                "instance": (
                    int(instance) if instance is not None else None
                ),
                "subject": (
                    subject_from_path(path)
                    or str(getattr(ds, "PatientID", "UNKNOWN"))
                ),
                "description": str(
                    getattr(ds, "SeriesDescription", "")
                ),
            }

            groups.setdefault(uid, []).append(record)

        except Exception as exc:
            errors.append({
                "path": str(path),
                "error": str(exc),
            })

    return groups, errors


def sort_slices(records):
    """Sort using the projection onto the slice normal."""
    first_orientation = next(
        (r["orientation"] for r in records if r["orientation"]),
        None,
    )

    if first_orientation and len(first_orientation) == 6:
        row = np.asarray(first_orientation[:3])
        col = np.asarray(first_orientation[3:])
        normal = np.cross(row, col)

        def key(record):
            pos = record["position"]
            if pos and len(pos) == 3:
                return (0, float(np.dot(pos, normal)))
            return (
                1,
                record["instance"]
                if record["instance"] is not None else 0,
            )

        return sorted(records, key=key)

    return sorted(
        records,
        key=lambda r: (
            r["instance"] is None,
            r["instance"] if r["instance"] is not None else 0,
            str(r["path"]),
        ),
    )


def validate_nifti(path: Path):
    """Validate that a NIfTI is readable and has valid 3D voxel data."""
    image = nib.load(str(path))
    shape = image.shape

    if len(shape) != 3 or any(d <= 0 for d in shape):
        raise ValueError(f"Expected non-empty 3D volume, got {shape}")

    spacing = image.header.get_zooms()[:3]
    if not all(np.isfinite(v) and v > 0 for v in spacing):
        raise ValueError(f"Invalid voxel spacing: {spacing}")

    if not np.all(np.isfinite(image.affine)):
        raise ValueError("NIfTI affine contains non-finite values")

    # Check data in chunks to avoid loading an entire large volume at once.
    proxy = image.dataobj
    if not np.isfinite(np.asarray(proxy[..., shape[2] // 2])).any():
        raise ValueError("Middle slice contains no finite values")

    if not np.any(np.isfinite(np.asarray(proxy[..., shape[2] // 2]))):
        raise ValueError("Middle slice contains no finite voxels")

    return shape, tuple(float(v) for v in spacing)


def convert_series(uid, records):
    """Convert one DICOM series to a compressed NIfTI file."""
    records = sort_slices(records)
    subject = records[0]["subject"] or "UNKNOWN"

    # Include a UID fragment to avoid filename collisions.
    safe_uid = re.sub(r"[^A-Za-z0-9]+", "_", uid)[-24:]
    filename = f"{subject}_{safe_uid}.nii.gz"
    output_path = OUTPUT_DIR / filename

    row = {
        "subject_id": subject,
        "series_instance_uid": uid,
        "series_description": records[0]["description"],
        "status": "FAILED",
        "number_of_slices": len(records),
        "dimensions": "",
        "voxel_spacing_mm": "",
        "output_filename": str(output_path.relative_to(ROOT)),
        "errors": "",
    }

    try:
        if len(records) < 2:
            raise ValueError(
                "Series has fewer than two slices; not treated as a 3D volume"
            )

        # SimpleITK reconstructs the volume from the ordered DICOM files
        # and retains physical spacing, origin, and direction.
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(r["path"]) for r in records])
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        image = reader.Execute()

        if image.GetDimension() != 3:
            raise ValueError(
                f"Expected 3D image, got {image.GetDimension()}D"
            )

        if any(v <= 0 for v in image.GetSize()):
            raise ValueError(f"Invalid image size: {image.GetSize()}")

        if output_path.exists() and not OVERWRITE_EXISTING:
            # Reuse only an existing file that passes validation.
            try:
                shape, spacing = validate_nifti(output_path)
            except Exception:
                output_path.unlink()
            else:
                row.update({
                    "status": "SKIPPED_EXISTING_VALID",
                    "dimensions": "x".join(map(str, shape)),
                    "voxel_spacing_mm": "x".join(
                        f"{v:.6g}" for v in spacing
                    ),
                })
                return row

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(output_path), useCompression=True)

        shape, spacing = validate_nifti(output_path)

        row.update({
            "status": "SUCCESS",
            "dimensions": "x".join(map(str, shape)),
            "voxel_spacing_mm": "x".join(
                f"{v:.6g}" for v in spacing
            ),
        })

    except Exception as exc:
        row["errors"] = str(exc)
        logger.exception("Failed to convert series %s", uid)

    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report_rows = []
    success = 0
    skipped = 0
    failed = 0

    try:
        groups, discovery_errors = discover_series(DICOM_ROOT)

        for error in discovery_errors:
            report_rows.append({
                "subject_id": subject_from_path(Path(error["path"])),
                "series_instance_uid": "",
                "series_description": "",
                "status": "DISCOVERY_ERROR",
                "number_of_slices": 0,
                "dimensions": "",
                "voxel_spacing_mm": "",
                "output_filename": "",
                "errors": f'{error["path"]}: {error["error"]}',
            })

        logger.info("Found %d distinct DICOM series", len(groups))

        for uid, records in sorted(groups.items()):
            row = convert_series(uid, records)
            report_rows.append(row)

            if row["status"] == "SUCCESS":
                success += 1
            elif row["status"] == "SKIPPED_EXISTING_VALID":
                skipped += 1
            else:
                failed += 1

    except Exception as exc:
        logger.exception("Conversion could not start: %s", exc)
        failed += 1
        report_rows.append({
            "subject_id": "",
            "series_instance_uid": "",
            "series_description": "",
            "status": "FATAL_ERROR",
            "number_of_slices": 0,
            "dimensions": "",
            "voxel_spacing_mm": "",
            "output_filename": "",
            "errors": str(exc),
        })

    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n========== DICOM CONVERSION SUMMARY ==========")
    print(f"Series processed: {len(report_rows)}")
    print(f"Converted successfully: {success}")
    print(f"Existing valid files skipped: {skipped}")
    print(f"Failed or discovery errors: {failed}")
    print(f"NIfTI output directory: {OUTPUT_DIR}")
    print(f"CSV report: {REPORT_PATH}")
    print("Original DICOM files were not modified.")
    print("==============================================")


if __name__ == "__main__":
    main()
