<div align="center">

CAMUS-LV-Research
From LV segmentation to functional assessment in echocardiography
Computer Vision Research · ASCEND Lab, NUST
</div>

Research use only. EchoLV is an assistive, non-diagnostic prototype. Its measurements and predictions are not clinically validated.

Overview
This project investigates a key challenge in automated echocardiography: good segmentation overlap does not always produce accurate left-ventricular (LV) measurements or ejection fraction (EF) estimates. Using the CAMUS dataset, the research examines how segmentation choices affect downstream geometry, EF-based classification, and application reliability—particularly when errors occur in end-systolic (ES) frames.
Research workflow

<sub>From CAMUS A4C ED/ES frames to LV segmentation, quantitative measurements, EF-based classification, robustness evaluation, and the EchoLV assistive application. The illustration groups robustness and application work in its final panel; these are Phase 4 and Phase 5, respectively. The dataset and preprocessing are prerequisites, not numbered research phases.</sub>
Dataset
Setting	Description
Dataset	CAMUS — Cardiac Acquisitions for Multi-structure Ultrasound Segmentation
View and frames	Apical four-chamber (A4C/4CH), ED and ES
Target	LV cavity segmentation
Project-specific split	400 training / 50 validation / 50 test patients, seed 42


Download source used: CAMUS-Human Heart Data on Kaggle. The dataset was originally developed by the CAMUS researchers; Kaggle is the download source, not the original scientific publisher. See the official CAMUS project and original publication.
The 400/50/50 split is specific to this research, not the official challenge split. Patient assignments are recorded in [`CAMUS_FINAL/splits/`](CAMUS_FINAL/splits/). Raw scans are not included in the repository.
Research phases
<details>
<summary><strong>Phase 1 — LV segmentation</strong></summary>

Controlled U-Net experiments investigated resolution (128, 256, 512), hyperparameters, network capacity, and loss functions. Later steps revisited downstream-sensitive errors and prepared the selected model for use in subsequent phases.
Explore: [Resolution](CAMUS_FINAL/phase1_segmentation/phase1A_resolution/) · [Hyperparameters](CAMUS_FINAL/phase1_segmentation/phase1B_hyperparameters/) · [Capacity](CAMUS_FINAL/phase1_segmentation/phase1C_capacity/) · [Loss](CAMUS_FINAL/phase1_segmentation/phase1D_loss/) · [Final validation](CAMUS_FINAL/phase1_segmentation/phase1E_final_validation/) · [Frozen model](CAMUS_FINAL/phase1_segmentation/phase1F_final_model/)
</details>

<details>
<summary><strong>Phase 2 — Quantitative LV analysis</strong></summary>

Predicted ED/ES masks are processed to estimate LV area, long-axis length, EDV, ESV, stroke volume (SV), and EF. The project analyzes how segmentation errors influence these derived measurements. Volumes are based on the project's single-view geometric approximation, not clinical volumetric ground truth.
Explore: [Measurement methodology](CAMUS_FINAL/phase2_quantitative/phase2A_measurement_method/) · [Final evaluation and error decomposition](CAMUS_FINAL/phase2_quantitative/phase2B_final_evaluation/)
</details>

<details>
<summary><strong>Phase 3 — EF-based classification</strong></summary>

Evaluates direct EF-based classification and a learned, feature-based classifier across lower, intermediate, and higher EF categories. The selected logistic-regression-based classifier achieved 70% held-out test accuracy and 0.663 macro-F1 in the reported evaluation.
Explore: [Label audit](CAMUS_FINAL/phase3_classification/phase3A_label_audit/) · [Baseline](CAMUS_FINAL/phase3_classification/phase3B_direct_baseline/) · [Model development](CAMUS_FINAL/phase3_classification/phase3C_classifier_development/) · [Final evaluation](CAMUS_FINAL/phase3_classification/phase3D_final_evaluation/)
</details>

<details>
<summary><strong>Phase 4 — Robustness and failure analysis</strong></summary>

Investigates error propagation, prediction confidence, image-quality subgroups, and representative failure cases. A central concern is that ES boundary errors can disproportionately affect derived volumes and EF, even when aggregate Dice scores appear strong.
Explore: [Error propagation](CAMUS_FINAL/phase4_robustness/phase4A_error_propagation/) · [Confidence](CAMUS_FINAL/phase4_robustness/phase4B_confidence_uncertainty/) · [Image quality](CAMUS_FINAL/phase4_robustness/phase4C_image_quality_subgroups/) · [Failure cases](CAMUS_FINAL/phase4_robustness/phase4D_representative_failure_cases/)
</details>

<details>
<summary><strong>Phase 5 — EchoLV assistive application</strong></summary>

Integrates research components into an assistive workflow for segmentation, LV measurements, EF-band interpretation, and quality/reliability feedback. Development includes system specifications, pipeline integration, application code, verification, and quality assessment. This is a research prototype, not a deployed clinical service.
Explore: [Application and launcher](CAMUS_FINAL/phase5_application/EchoLV_Deployment/) · [Development notebooks](CAMUS_FINAL/phase5_application/phase5_Application_Code_Files/) · [Phase 5 artifacts](CAMUS_FINAL/phase5_application/)
</details>

Results at a glance
Evaluation	Reported result
EF-based classifier · held-out test accuracy	70%
EF-based classifier · macro-F1	0.663


These are research-set evaluation results, not clinical performance claims. See Phase 3 for methodology.
Research insights
Observation	Research significance
High Dice alone is insufficient	Segmentation must also be evaluated through downstream measurements.
ES errors are consequential	Small boundary deviations may propagate into ESV and EF estimates.
Classification is imperfect	Reported test result: 70% accuracy, 0.663 macro-F1.
Quality and confidence matter	Failure analysis helps identify when outputs require caution.


Results reflect the project's experimental split and do not establish clinical generalization. Detailed metrics and plots are retained in the corresponding phase folders; early baselines should not be confused with final evaluations.
Repository layout
CAMUS_FINAL/
├── configs/
├── splits/
├── phase1_segmentation/
├── phase2_quantitative/
├── phase3_classification/
├── phase4_robustness/
└── phase5_application/
Reproducibility
1. Obtain CAMUS separately from an authorized source and review the [saved patient splits](CAMUS_FINAL/splits/).
2. Open the relevant phase notebook in Google Colab or a compatible Python environment.
3. Update its Google Drive paths and install the dependencies imported by that notebook. The application has a separate [requirements file](CAMUS_FINAL/phase5_application/EchoLV_Deployment/requirements.txt).
4. Run phases in dependency order: segmentation → measurements → classification → reliability analysis → application integration.
Note: Raw images, trained checkpoints, and large intermediate arrays are excluded from Git. Some experiments require retraining or regenerating artifacts. The repository is a research record rather than a one-command reproducible package.
Limitations and responsible use
Single-view geometric EF estimates, dataset-specific evaluation, segmentation-to-measurement error propagation, and imperfect confidence estimates limit interpretation. The system is non-diagnostic and must not be used to guide patient care. Review notebook outputs and data-use permissions before publicly redistributing CAMUS-derived material.
Acknowledgments and citation
Developed as a computer-vision research internship project at ASCEND Lab, National University of Sciences and Technology (NUST), Pakistan.
Original CAMUS publication: Leclerc, S., et al. (2019). Deep Learning for Segmentation Using an Open Large-Scale Dataset in 2D Echocardiography. IEEE Transactions on Medical Imaging, 38(9), 2198–2210. DOI: 10.1109/TMI.2019.2900516.
Dataset download source: Kaggle — CAMUS-Human Heart Data.
License
No open-source license has been added. Project-specific materials are not licensed for reuse unless permission is granted; third-party datasets and software retain their respective terms.
