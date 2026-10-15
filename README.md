# Alzheimer's Hippocampus Segmentation

A research-oriented project for segmenting the hippocampus from brain MRI scans in the context of Alzheimer's disease analysis. The repository is organized around a medical imaging pipeline using Python, PyTorch, and established neuroimaging libraries for preprocessing, model development, and evaluation.

> Status: The repository currently includes dataset artifacts, preprocessing outputs, QC reports, and project documentation. The core model-training and app layers are scaffolded and are intended for future implementation.

## Project goal

The main objective is to build a reproducible pipeline that:

- ingests MRI volumes and supporting metadata,
- prepares and validates 3D brain images,
- creates aligned segmentation targets,
- trains a U-Net model for hippocampus segmentation,
- evaluates segmentation quality,
- and produces measurement outputs that can support downstream clinical or research analysis.

This project is intended for research and educational use only and is not a clinical diagnostic system.

## Current repository status

The current workspace contains:

- MRI data organized under `dataset/`
- preprocessed volumes in `dataset/preprocessed/`
- train/validation/test split folders in `dataset/splits/`
- generated reports and plots in `results/`
- documentation files in `documentation/`
- Python and model scaffolding under `models/`, `training/`, `preprocessing/`, and `app/`

At the moment, the repository is best understood as a project foundation and data-processing workspace rather than a fully finished application. The dataset preparation and QC workflow are the clearest implemented parts of the project.

## Pipeline overview

1. Acquire brain MRI data and review metadata.
2. Convert DICOM series to NIfTI format when applicable.
3. Validate image orientation, spacing, and anatomy.
4. Prepare and align hippocampus masks.
5. Split the dataset by subject to reduce leakage.
6. Train a 3D or 2D U-Net segmentation model.
7. Run inference on new scans.
8. Evaluate predictions with Dice, IoU, precision, recall, and volumetric checks.
9. Export reports, visuals, and measurement summaries.

## Technologies

- Python
- PyTorch
- NumPy
- SciPy
- SimpleITK
- NiBabel
- OpenCV
- Matplotlib
- Pandas
- scikit-learn
- Jupyter or Python-based development workflow

## Repository structure

```text
Alzheimer-Hippocampus-Segmentation/
|-- README.md
|-- requirements.txt
|-- main.py
|-- app/
|   |-- app.py
|-- models/
|   |-- model_config.py
|   |-- unet.py
|-- preprocessing/
|   |-- dataset_split.py
|   |-- dicom_to_nifti.py
|   |-- image_preprocessing.py
|   |-- mask_preparation.py
|   |-- normalization.py
|   |-- resampling.py
|   |-- verify_nifti.py
|-- training/
|   |-- evaluate.py
|   |-- metrics.py
|   |-- train.py
|-- dataset/
|   |-- raw/
|   |-- nifti/
|   |-- preprocessed/
|   |-- splits/
|       |-- train/
|       |-- validation/
|       |-- test/
|-- results/
|   |-- checkpoints/
|   |-- plots/
|   |-- predictions/
|   |-- segmentation/
|-- documentation/
|   |-- dataset_statistics.md
|   |-- preprocessing_report.md
|   |-- project_notes.md
|-- .gitignore
```

## Data organization

The project expects MRI data to be stored in a structured dataset layout similar to the following:

- `dataset/raw/` - original source files
- `dataset/nifti/` - NIfTI-converted volumes
- `dataset/preprocessed/` - normalized and processed images
- `dataset/splits/` - train/validation/test partitions

Large imaging datasets and generated outputs should not be committed to Git unless there is a documented storage and sharing plan.

## Setup

1. Clone the repository.
2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate it:

- Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

- Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Confirm the dataset folder structure matches the expected MRI workflow before running preprocessing or training scripts.

## Development notes

This project is structured to support a typical deep-learning medical imaging pipeline:

- preprocessing and QC modules prepare data for training,
- model files define the core segmentation architecture,
- training scripts handle training and evaluation,
- app files represent the user-facing interface layer,
- and documentation captures dataset status and results.

The codebase is intentionally organized so new components can be added as the segmentation model and application layers mature.

## Key considerations

- Data quality and alignment are critical for hippocampus segmentation.
- MRI preprocessing must be consistent between training and inference.
- Subject-level splitting should be used to avoid leakage.
- Model validation should include both segmentation metrics and volumetric sanity checks.
- The output should remain research focused and not interpreted as clinical-grade diagnosis.

## Limitations

- The repository is still in a project-setup and pipeline-development stage.
- The model implementation and production-ready app layer are not yet complete.
- Dataset access and licensing must be handled according to the relevant medical-data policies.
- Generated reports should be reviewed before being used for scientific or clinical conclusions.

## Disclaimer

This repository is intended for academic, research, and educational use. It is not a substitute for medical advice, and it should not be used to make clinical decisions.

## License and usage

This project does not currently include a formal project-specific license file. Review repository policies before reuse or redistribution, and respect all dataset access restrictions and institutional requirements.
