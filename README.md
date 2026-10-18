# Alzheimer's Disease Detection Using Hippocampus Segmentation with U-Net

An AI-assisted research project that aims to segment the hippocampus from brain MRI scans using a 3D U-Net model and provide analysis through an Android application. The planned mobile app will communicate with a Python backend and may be distributed through the Google Play Store after implementation, testing, and approval.

> **Project status:** Dataset preprocessing, mask preparation, subject-level splitting, U-Net architecture setup, smoke-test training (10 epochs), and validation scripts are now implemented. Formal training on the full dataset, held-out evaluation, backend implementation, Android app development, and Play Store publication are still pending. This is a research prototype, not a medical diagnostic device.

## Table of Contents

| No. | Section | Description |
|---:|---|---|
| 1 | [Project Overview](#project-overview) | Introduction and project status |
| 2 | [Objectives](#objectives) | Main project goals |
| 3 | [System Architecture](#system-architecture) | Planned system modules and connections |
| 4 | [Technology Stack](#technology-stack) | Frameworks, tools, and technologies |
| 5 | [Programming Languages Used and Planned](#programming-languages-used-and-planned) | Languages used or planned for development |
| 6 | [Dataset and Current Progress](#dataset-and-current-progress) | Dataset details and completed preprocessing work |
| 7 | [Project Structure](#project-structure) | Repository organization |
| 8 | [Team Responsibilities](#team-responsibilities) | Work allocation among four members |
| 9 | [Member-wise and Overall Project Completion](#member-wise-and-overall-project-completion) | Provisional progress estimates for each member and the project |
| 10 | [Development Roadmap](#development-roadmap) | Planned implementation stages |
| 11 | [Setup](#setup) | Environment setup instructions |
| 12 | [Planned Application Workflow](#planned-application-workflow) | Intended app usage flow |
| 13 | [Evaluation Plan](#evaluation-plan) | Model evaluation measures |
| 14 | [Privacy and Security](#privacy-and-security) | Data protection considerations |
| 15 | [Limitations](#limitations) | Current constraints |
| 16 | [Future Enhancements](#future-enhancements) | Possible future work |
| 17 | [Acknowledgment](#acknowledgment) | Dataset and project acknowledgments |
| 18 | [Disclaimer](#disclaimer) | Research-use notice |

## Project Overview

Alzheimer's disease is associated with changes in brain structure, including the hippocampus. This project explores the use of deep learning to segment the hippocampus from T1-weighted brain MRI scans. The intended system combines MRI data conversion and preprocessing, 3D U-Net segmentation, evaluation and measurement, a Python backend, and an Android application.

The system is being developed as a research and educational project. It must not be used to make clinical decisions.

## Objectives

- Organize and validate ADNI MRI data.
- Convert DICOM image series to NIfTI volumes.
- Develop a reproducible MRI preprocessing pipeline.
- Obtain and verify hippocampus segmentation masks.
- Train and evaluate a 3D U-Net segmentation model.
- Calculate hippocampal measurements from verified masks.
- Develop backend APIs to connect the model and mobile app.
- Build an Android app for MRI upload, result visualization, and scan history.
- Prepare the app for testing and possible Google Play Store distribution.

## System Architecture

### High-level system architecture

```mermaid
flowchart TB
    subgraph DATA[1. Data acquisition]
        A[ADNI dataset access] --> B[Baseline T1-weighted MRI]
        B --> C[Original DICOM series]
    end

    subgraph PREP[2. MRI preprocessing - Python / VS Code]
        C --> D[Inspect series and metadata]
        D --> E[Convert DICOM to NIfTI]
        E --> F[Validate dimensions, orientation and voxel spacing]
        F --> G[Quality control and slice visualization]
        G --> H[Percentile-based normalization / resampling]
        H --> I[Prepared MRI volumes]
    end

    subgraph LABEL[3. Ground-truth preparation]
        M[Obtain authorized hippocampus masks] --> N[Check labels and image-mask alignment]
        N --> O[Pair MRI volumes with masks]
        O --> P[Subject-level train / validation / test split]
    end

    I --> O

    subgraph AI[4. Deep learning - PyTorch / MONAI]
        P --> Q[Load MRI-mask pairs via CacheDataset]
        Q --> R[3D U-Net encoder, bottleneck and decoder]
        R --> S[Train with DiceCELoss and Adam optimizer]
        S --> T[Save checkpoint per epoch]
        T --> U[Inference: predict hippocampus mask via sliding-window]
    end

    subgraph EVAL[5. Evaluation and measurements - Python]
        U --> V[Compare prediction with reference mask]
        V --> W[Dice / IoU / precision / recall]
        U --> X[Validate image geometry and mask]
        X --> Y[Calculate left and right hippocampal volume]
        Y --> Z[Generate measurements and analysis results]
    end

    subgraph SERVER[6. Backend - FastAPI]
        API[REST API] --> AUTH[Authentication and role checks]
        AUTH --> UP[Validate and receive MRI upload]
        UP --> PROC[Run preprocessing and model inference]
        PROC --> RES[Return segmentation and measurements]
        API --> HIST[User scan history and report endpoints]
    end

    U --> PROC
    Z --> RES

    subgraph DB[7. Database - MySQL]
        DBU[Users and roles]
        DBS[Scan metadata]
        DBR[Analysis results and reports]
        DBL[Activity and model-version logs]
    end
    AUTH <--> DBU
    UP <--> DBS
    RES <--> DBR
    HIST <--> DBS
    HIST <--> DBR
    API <--> DBL

    subgraph MOBILE[8. Android app - Flutter / Dart / Android Studio]
        APP[Android application]
        LOGIN[Registration and login]
        DASH[User dashboard]
        FILE[Select and upload MRI]
        VIEW[View MRI and segmentation overlay]
        MEAS[View measurements and report]
        HISTORY[Personal scan history]
        ADMIN[Authorized admin screens]
        APP --> LOGIN --> DASH
        DASH --> FILE
        DASH --> HISTORY
        DASH --> ADMIN
        RES --> VIEW
        RES --> MEAS
    end

    FILE --> API
    API --> VIEW
    API --> MEAS
    HIST --> HISTORY

    subgraph RELEASE[9. Testing and distribution]
        TEST[Integration, security and device testing]
        AAB[Signed Android App Bundle]
        CONSOLE[Google Play Console: testing, declarations and review]
        USERS[Release to users after approval]
        APP --> TEST --> AAB --> CONSOLE --> USERS
    end
```

### Detailed module descriptions

| Module | Input | Processing | Output |
|---|---|---|---|
| Dataset acquisition | ADNI MRI data | Organize scans by subject and series; follow dataset access conditions | Original MRI files and metadata |
| DICOM conversion | DICOM series | Read image series and convert to NIfTI while checking spatial metadata | NIfTI volumes |
| Preprocessing and QC | NIfTI volumes | Validate dimensions/orientation/spacing; percentile-based normalization; duplicate detection via SHA-256 | Validated, normalized MRI volumes and QC reports |
| Mask preparation | Authorized hippocampus labels | Confirm label meaning, geometry and alignment; pair masks with MRI | Verified MRI-mask pairs |
| Dataset splitting | Verified pairs | 70/15/15 subject-level split (train/validation/test) — seeded for reproducibility | `train_val_test_split.csv` |
| U-Net training | Training pairs | 3D U-Net with `DiceCELoss`, `Adam (lr=1e-4)`, `RandCropByPosNegLabeld` patches (96×96×96); per-epoch checkpoints | Trained segmentation model checkpoints |
| Inference | New, supported MRI volume | Apply the same preprocessing and run sliding-window inference (overlap=0.25) | Predicted hippocampus mask |
| Evaluation | Predictions and reference masks | Calculate Dice, IoU, precision and recall; inspect errors | Evaluation metrics and plots |
| Measurement | Validated mask and voxel geometry | Calculate left/right volumes and related measurements | Quantitative results |
| FastAPI backend | App requests and uploaded files | Authenticate, authorize, validate, invoke processing and return results | API responses and controlled access |
| MySQL database | Account and analysis metadata | Store records with ownership and access controls | Persistent application records |
| Android app | User input and API responses | Handle login, upload, visualization, history and reports | Mobile user experience |
| Release pipeline | Tested app build | Sign bundle, complete Play Console declarations/testing/review | Published app if approved |

> **Important:** This diagram represents the target architecture. Several modules—especially backend, database, and Android app—are still planned and must not be considered implemented until tested.

## Technology Stack

| Component | Planned technology | Purpose |
|---|---|---|
| MRI dataset | ADNI | Source of brain MRI data |
| Image format | DICOM, NIfTI | MRI input and processing |
| Development environment | VS Code | Python and backend development |
| Image processing | SimpleITK, NiBabel, NumPy | Reading and processing MRI volumes |
| Data analysis | Pandas, Matplotlib | Reports and visualizations |
| Deep learning | PyTorch, MONAI | 3D U-Net implementation and training |
| Backend | Python, FastAPI | APIs and model inference |
| Database | MySQL | User, scan, and analysis records |
| Mobile app | Flutter, Dart, Android Studio | Android user interface |
| Version control | Git, GitHub | Collaboration and source management |
| Distribution | Android App Bundle, Google Play Console | App release, subject to requirements and review |

## Programming Languages Used and Planned

| Language | Where it is used | Purpose | Status |
|---|---|---|---|
| **Python** | MRI preprocessing, 3D U-Net model, training/validation scripts, FastAPI backend | Image processing, deep learning, data analysis, API services | Used for preprocessing, model training setup, and validation; backend planned |
| **Dart** | Flutter Android application | Mobile screens, navigation, API communication and result display | Planned for app development |
| **SQL** | MySQL database | Create and query user, scan, analysis, report and activity records | Planned for backend/database implementation |
| **PowerShell** | Windows development terminal | Create environments, install packages, run scripts and manage the project | Used for local development commands |
| **Markdown** | `README.md` and project documentation | Document setup, architecture, progress and usage | In use |
| **Mermaid** | Architecture diagram in this README | Describe the system workflow and module relationships | Used in documentation |
| **YAML** | Potential configuration and CI files | Store structured configuration or automation workflows, if introduced | Optional / not confirmed as implemented |

**Primary programming languages:** Python for AI and backend development, Dart for the Android application, and SQL for database operations. Markdown and Mermaid are used for documentation, while PowerShell supports the Windows development workflow.

## Dataset and Current Progress

The project uses baseline T1-weighted MRI data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). Follow the dataset provider's access conditions and data-use agreement.

### Confirmed results

| Item | Current result |
|---|---|
| DICOM files identified | 830 |
| MRI volumes converted in the test subset | 5 |
| Successful conversions in that subset | 5 |
| Conversion failures in that subset | 0 |
| Typical converted volume dimensions | 256 × 256 × 166 |
| Approximate voxel spacing | 1.016 × 1.016 × 1.2 mm |
| Initial QC volumes checked | 5 |
| QC warnings/rejections in the reported test run | 0 |
| Preprocessing pipeline | Percentile normalization (p1–p99) with duplicate detection |
| Dataset split | 70 / 15 / 15 subjects (train / validation / test) — seed 42 |
| U-Net architecture | 3D U-Net via MONAI (channels: 16-32-64-128, strides: 2-2-2, res units: 2) |
| Loss function | DiceCELoss (one-hot, softmax) |
| Optimizer | Adam, lr = 1e-4 |
| Smoke-test training epochs completed | 10 |
| Per-epoch checkpoints saved | 10 (`unet_epoch_1.pth` … `unet_epoch_10.pth`) |
| Validation script | `training/validate_unet.py` — sliding-window inference, Dice scoring |
| Prediction diagnostics script | `training/diagnose_predictions.py` — voxel counts, foreground probability |

> **Note:** The 5-volume conversion/QC test does not mean the complete dataset has been preprocessed. Smoke-test training used a small subject pool to verify the pipeline end-to-end. Formal training on the full 100-subject dataset and held-out evaluation have not yet been completed.

### Existing reports and visualizations

- `results/dicom_conversion_report.csv`
- `results/preprocessing_qc_report_*.csv`
- `results/plots/` — MRI slice quality-control plots
- `results/training/` — per-epoch model checkpoints (`unet_epoch_1.pth` … `unet_epoch_10.pth`)

Large raw datasets, private data, virtual environments, generated caches, and model weights should not be committed to Git unless the project has an approved storage and sharing plan.

## Project Structure

```text
Alzheimer-Hippocampus-Segmentation/
├── preprocessing/
│   ├── dicom_to_nifti.py           # DICOM → NIfTI conversion
│   ├── image_preprocessing.py      # Initial QC checks
│   ├── preprocess_mri.py           # Percentile normalization pipeline
│   ├── normalize_mri.py
│   ├── normalization.py
│   ├── resampling.py
│   ├── harp_registration.py        # HARP label registration
│   ├── mask_preparation.py         # (planned) mask pairing
│   ├── create_dataset_split.py     # 70/15/15 subject-level split
│   ├── dataset_split.py
│   ├── check_final_geometry.py
│   ├── check_final_labels.py
│   ├── check_label_locations.py
│   ├── check_patch_balance.py
│   ├── generate_label_qc_all.py
│   ├── inspect_mri_intensity.py
│   ├── verify_dataset_split.py
│   ├── verify_nifti.py
│   ├── verify_normalized_mri.py
│   ├── visualize_mri.py
│   └── test_unet_dataloader.py
├── models/
│   ├── unet.py                     # (placeholder)
│   └── model_config.py             # (placeholder)
├── training/
│   ├── train_unet.py               # 3D U-Net smoke-test training (10 epochs)
│   ├── validate_unet.py            # Sliding-window validation + Dice scoring
│   ├── diagnose_predictions.py     # Voxel count / probability diagnostics
│   ├── train.py
│   ├── train_unet_backup.py
│   └── evaluate.py                 # (placeholder)
│   └── metrics.py                  # (placeholder)
├── app/
│   └── app.py                      # (placeholder)
├── dataset/
│   ├── adni_downloads/
│   ├── final/
│   │   ├── normalized_images/
│   │   ├── labels/
│   │   └── metadata/               # train_val_test_split.csv
│   ├── harp_labels/
│   ├── masks/
│   ├── metadata/
│   ├── nifti/
│   ├── preprocessed/
│   ├── processed/
│   ├── raw/
│   └── splits/
├── results/
│   ├── training/                   # unet_epoch_1.pth … unet_epoch_10.pth
│   ├── checkpoints/
│   ├── label_qc_all/
│   ├── plots/
│   ├── predictions/
│   ├── registration_test/
│   ├── segmentation/
│   ├── visualizations/
│   ├── dicom_conversion_report.csv
│   └── preprocessing_qc_report_*.csv
├── documentation/
│   ├── dataset_statistics.md
│   ├── preprocessing_report.md
│   └── project_notes.md
├── .venv/
├── .gitignore
├── requirements.txt
└── README.md
```

Keep original MRI data outside the repository or in an approved private data store.

## Team Responsibilities

The work is divided into four approximately equal responsibility areas. The percentages refer to planned workload, not a claim that every member has completed that share.

| Member | Responsibility | Main tasks | Deliverables |
|---|---|---|---|
| Member 1 | Dataset collection and MRI preprocessing | Collect and organize ADNI data; convert DICOM to NIfTI; validate dimensions and metadata; perform normalization, resampling, and other suitable preprocessing; prepare and verify masks; create subject-level data splits; generate QC reports. | Organized dataset, conversion/preprocessing scripts, verified MRI-mask pairs, QC reports, split files. |
| Member 2 | U-Net model development and training | Implement 3D U-Net via MONAI; build data loaders; apply DiceCELoss and augmentation; train and validate; save checkpoints and training history; generate predictions; provide inference function. | U-Net training pipeline, smoke-test checkpoints, validation + diagnostics scripts, inference interface. |
| Member 3 | Evaluation, hippocampal analysis, and backend | Evaluate segmentation using appropriate metrics; analyze errors; calculate hippocampal volumes and asymmetry; build FastAPI endpoints; connect MySQL; integrate the trained model; implement authentication and access controls; test APIs. | Evaluation report, measurement module, backend APIs, database integration, API documentation. |
| Member 4 | Android app, integration, and release preparation | Develop Flutter screens; implement login, upload, result display, history, and reports; connect APIs; create app icon; test on Android devices; build signed AAB; prepare store listing and testing materials. | Android app, app icon, integrated interface, tested release bundle, Play Store listing materials. |

All members contribute to weekly meetings, project diaries, code reviews, integration testing, final documentation, review presentations, and the final demonstration.


## Member-wise and Overall Project Completion

The percentages below are **provisional progress estimates** based on the work completed and implementation status currently documented in this repository. They are not automatically calculated from Git commits or a formal assessment. Planning and study are acknowledged, but are not counted as completed implementation.

| Member | Main responsibility | Estimated completion | Current status |
|---|---|---:|---|
| Member 1 | ADNI preprocessing, HarP registration, normalization, dataset split | 60% | ADNI data organization, DICOM-to-NIfTI conversion, initial QC, percentile normalization pipeline, HARP registration, multiple QC/verification scripts, and dataset split (70/15/15) implemented. Ground-truth mask pairing and full-dataset preprocessing run remain pending. |
| Member 2 | 3D U-Net, MONAI dataloader, training, validation, diagnostics | 40% | 3D U-Net configured via MONAI; DiceCELoss + Adam optimizer set up; smoke-test training (10 epochs, 10 checkpoints saved); sliding-window validation script and prediction diagnostics script implemented. Full training run and formal evaluation pending. |
| Member 3 | Model improvement, tuning, augmentation, segmentation refinement | 0% | Model tuning, augmentation strategy, and segmentation refinement have been planned; implementation remains pending. |
| Member 4 | Final evaluation, comparison, visualization, integration/report | 0% | Evaluation metrics, result visualization, model comparison, and final integration report have been planned; implementation remains pending. |
| **Overall project** | **Complete project** | **~25%** | **Estimated average of the four member completion percentages.** |

**Calculation:** (60% + 40% + 0% + 0%) ÷ 4 = **25%**.

### Project Completion Graph

The following **Mermaid chart is embedded directly in this README code file**; no separate image is required. GitHub renders the chart when it displays the Markdown file.

```mermaid
xychart-beta
    title "Project completion by member"
    x-axis [Member 1, Member 2, Member 3, Member 4, Overall]
    y-axis "Completion (%)" 0 --> 100
    bar [60, 40, 0, 0, 25]
```

*Each bar represents progress out of 100%. The overall figure is the average of the four members' estimates.*

> **Note:** This is an approximate planning figure, not a verified measurement of all project tasks. Update the table as implementation milestones are completed. Smoke-test training does not mean that formal training on the full dataset or held-out evaluation has been completed.

> **⚠️ Important:** The 40% for Member 2 means the training pipeline is implemented, not that the model is performing well. The current 10-epoch validation Dice is only approximately **0.0031**, so model improvement is still a major remaining task.

## Development Roadmap

| Phase | Activities | Status |
|---|---|---|
| 1. Research and setup | Topic selection, literature survey, repository, environment, project planning | ✅ Completed |
| 2. Dataset acquisition and conversion | ADNI data organization, DICOM-to-NIfTI conversion, initial validation | ✅ Completed |
| 3. Full MRI preprocessing | Normalization, resampling, additional suitable processing, QC | 🟡 Partially completed |
| 4. Mask preparation | Obtain masks, verify alignment and labels, pair with MRI, subject-level splits | 🟡 Partially completed |
| 5. U-Net development | Implement, train, validate, save model, generate predictions | 🟡 Partially completed |
| 6. Evaluation and analysis | Dice, IoU, precision, recall, error analysis, volume measurements | 🟡 Started |
| 7. Backend | FastAPI, MySQL, authentication, upload and inference endpoints | ⏳ Pending |
| 8. Android application | Flutter UI, API integration, visualization, history, reports | ⏳ Pending |
| 9. Testing and release | Integration/device tests, privacy documentation, signed AAB, closed testing, Play Store review | ⏳ Pending |

## Setup

### Prerequisites

- Windows 10/11 or a compatible development environment.
- Python version compatible with the selected PyTorch, MONAI, and image-processing packages.
- VS Code, Git, Android Studio, and Flutter SDK for mobile development.
- Access to ADNI data under its applicable terms.
- MySQL for the planned backend.

### Clone the repository

```powershell
git clone https://github.com/minnachirakkal775-bot/Alzheimer-Hippocampus-Segmentation.git
cd Alzheimer-Hippocampus-Segmentation
```

### Create and activate a Python virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies from the maintained requirements file:

```powershell
pip install -r requirements.txt
```

If a requirements file has not yet been created, install and pin the dependencies used by implemented scripts before sharing the environment. Check package compatibility rather than assuming arbitrary versions will work together.

### Run the existing conversion script

After placing authorized DICOM data in the configured input location and checking the script's path settings:

```powershell
python preprocessing/dicom_to_nifti.py
```

### Run the MRI preprocessing / normalization script

```powershell
python preprocessing/preprocess_mri.py
```

### Create the subject-level dataset split

Ensure 100 valid MRI-mask pairs exist under `dataset/final/` before running:

```powershell
python preprocessing/create_dataset_split.py
```

This produces `dataset/final/metadata/train_val_test_split.csv` with a 70/15/15 split (seed 42).

### Run the U-Net smoke-test training

```powershell
python training/train_unet.py
```

Checkpoints are saved to `results/training/` after each epoch.

### Run validation on the saved checkpoint

```powershell
python training/validate_unet.py
```

Reports mean/median/min/max Dice scores across all validation subjects.

### Run prediction diagnostics

```powershell
python training/diagnose_predictions.py
```

Prints per-subject voxel counts and foreground probability statistics.

### Run the existing quality-checking script

```powershell
python preprocessing/image_preprocessing.py
```

Review the generated CSV reports and plots.

### Mobile app setup

After the Flutter project has been created:

```powershell
flutter doctor
flutter pub get
flutter run
```

Configure the API base URL to point to a backend reachable from the emulator or physical device. `localhost` on a phone does not automatically refer to the development computer.

## Planned Application Workflow

1. A user signs in to the Android application.
2. The user selects an MRI file in a supported format.
3. The app sends the file securely to the backend.
4. The backend validates the file and runs the configured preprocessing and model inference pipeline.
5. The 3D U-Net model generates a hippocampus segmentation via sliding-window inference.
6. Measurements are calculated only after the mask and image geometry have been validated.
7. The backend stores authorized scan and analysis records.
8. The app displays the image, segmentation overlay, measurements, and report.
9. Users can view their own scan history. Administrative access must be role-controlled and logged.

## Evaluation Plan

The segmentation model will be evaluated against verified ground-truth masks using held-out subjects. Planned metrics include:

- **Dice similarity coefficient:** overlap between predicted and reference masks.
- **Intersection over Union (IoU):** intersection divided by union of predicted and reference regions.
- **Precision:** proportion of predicted positive voxels that are correct.
- **Recall:** proportion of reference positive voxels recovered by the model.

The validation script (`training/validate_unet.py`) already implements Dice scoring and will be extended for the full metric suite. Report the evaluation split, sample count, metric definitions, and limitations. Do not report performance values until the corresponding experiments have been run and checked.

## Privacy and Security

MRI data can be sensitive. Before making the app available to other users:

- Follow ADNI data-use terms and do not redistribute restricted data.
- Collect only information required by the application.
- Use secure transport for uploads and API requests.
- Enforce authentication, role-based permissions, and record ownership on the backend.
- Do not rely only on hiding controls in the mobile interface for access control.
- Restrict and log administrative access to sensitive records.
- Define secure storage, retention, deletion, backup, and incident-handling procedures.
- Publish an accurate privacy policy and complete applicable Google Play Data safety declarations.
- Use synthetic or otherwise authorized test data during development and demonstrations.

## Limitations

- Initial conversion and QC results cover only five MRI volumes.
- Full-dataset preprocessing and mask pairing have not yet been run end-to-end.
- U-Net smoke-test training (10 epochs) verified the pipeline but is not formal model training.
- Held-out evaluation metrics (Dice, IoU, precision, recall) have not yet been computed.
- Segmentation performance and clinical validity are therefore unknown.
- Alzheimer's stage classification would require suitable labels, adequate data, and separate validation; it is not established by hippocampus segmentation alone.
- The Android app, backend, and Play Store release are planned work, not completed deliverables.

## Future Enhancements

- Complete and validate the full preprocessing pipeline on all available subjects.
- Run formal training on the full 100-subject dataset with early stopping.
- Train and compare baseline 3D U-Net with suitable variants such as Attention U-Net, if resources and data permit.
- Add robust 2D/3D visualization and longitudinal analysis where repeat scans are available.
- Explore classification only with appropriate labels and rigorous validation.
- Add explainability features with clear limitations.
- Improve accessibility, usability, and multilingual support.
- Conduct broader testing and prepare a secure production deployment.

## Acknowledgment

The project uses MRI data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), subject to its access requirements and data-use conditions. The team acknowledges the researchers, institutions, participants, and supporting organizations involved in making the dataset available. The 3D U-Net implementation uses the MONAI (Medical Open Network for AI) framework.

## Disclaimer

This application is an academic research prototype. It is not intended to diagnose, treat, cure, or prevent Alzheimer's disease or any other medical condition. Outputs must not replace assessment by qualified healthcare professionals. No claim of clinical accuracy or reliability should be made without appropriate independent validation and regulatory review.
