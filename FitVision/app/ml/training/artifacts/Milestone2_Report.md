# Machine Learning (CS 363) — Course Project
## Milestone 2: Model Training & Evaluation Report

**Project Name:** FitVision
**Course Code:** CS363 — Machine Learning
**Institution:** October University for Modern Science and Arts (MSA University)
**Faculty:** Faculty of Computer Science
**Instructor:** Assoc. Prof. Nermin Abd El-Wahab, Dr. Ebtsam El-Hussany
**Teaching Assistants:** Eng. Omnia Fawzy, Eng. Youssef Araby, Eng. Mazen Ashraf, Eng. Ahmed Yasser
**Submission Date:** May 2026

---

### Team Members

| Name | Student ID |
| :--- | :--- |
| Omar Hesham | 247469 |
| Mahmoud Abdelkareem | 247519 |
| Ahmed Salah | 247841 |
| Ahmed Ebrahim | 246057 |
| Noureen Mohamed | 246173 |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Dataset Description](#3-dataset-description)
4. [Data Preprocessing & Feature Engineering](#4-data-preprocessing--feature-engineering)
5. [Model Selection & Justification](#5-model-selection--justification)
6. [Evaluation Strategy & Metrics](#6-evaluation-strategy--metrics)
7. [Results & Comparative Analysis](#7-results--comparative-analysis)
8. [Error Analysis & Critical Reflection](#8-error-analysis--critical-reflection)
9. [Conclusion & Future Work](#9-conclusion--future-work)
10. [Teamwork & Task Delegation](#10-teamwork--task-delegation)

---

## 1. Introduction

Physical exercise is a cornerstone of public health, yet poor form and lack of real-time feedback remain leading causes of workout-related injuries. FitVision addresses this gap by using computer vision and machine learning to automatically classify exercises from video input and evaluate movement quality — all without specialised hardware.

**Problem Statement:** Most people who exercise without a personal trainer cannot receive immediate feedback on whether they are performing an exercise correctly. Existing solutions either require expensive wearable sensors or rely on cloud-based processing with significant latency.

**Objective:** Build a lightweight, server-side ML pipeline that (1) classifies the type of exercise being performed in a video, and (2) evaluates form quality in real time, using only a standard camera feed.

**Dataset:** A custom dataset of exercise videos was collected and processed through Google's MediaPipe Pose framework. Joint-distance features were extracted for four exercise classes: Push-Up, Squat, Jumping Jack, and Pull-Up, yielding 48 engineered features per frame.

**Models Implemented:** Three models from two distinct algorithmic families were trained and compared:
- Logistic Regression (linear baseline)
- Random Forest (ensemble tree-based)
- XGBoost (gradient-boosted tree)

**Key Result:** XGBoost achieved the best generalisation performance, with a Cross-Validated Macro F1-Score of **82.37%** on unseen videos, using a leakage-free video-level evaluation strategy.

---

## 2. Literature Review

### 2.1 BlazePose: On-device Real-time Body Pose Tracking

**Reference:** Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020). *BlazePose: On-device Real-time Body Pose tracking.* CVPR Workshop on Computer Vision for Augmented and Virtual Reality.

| Aspect | Details |
| :--- | :--- |
| **Problem** | Real-time human pose estimation on mobile devices with limited compute |
| **Dataset** | Internal dataset of ~50,000 annotated images across diverse environments |
| **Model** | Two-stage CNN: a detector followed by a regression-based landmark network |
| **Performance** | Real-time inference at 30+ FPS on mobile CPU; 33 body landmarks with depth estimation |
| **Limitations** | Single-person only; accuracy degrades with severe occlusion or extreme viewpoints |
| **Relation to FitVision** | FitVision's entire feature extraction pipeline is built on top of MediaPipe Pose, which is the production implementation of BlazePose. Understanding its 33-landmark output is fundamental to our 48 joint-distance features. |

### 2.2 Human Activity Recognition Using Wearable Sensors and Machine Learning

**Reference:** Bulling, A., Blanke, U., & Schiele, B. (2014). *A tutorial on human activity recognition using body-worn inertial sensors.* ACM Computing Surveys, 46(3), 1–33.

| Aspect | Details |
| :--- | :--- |
| **Problem** | Recognising human activities (walking, running, sitting) from body-worn accelerometer and gyroscope data |
| **Dataset** | Multiple benchmark datasets including OPPORTUNITY and PAMAP2 |
| **Models** | Decision Trees, SVMs, Hidden Markov Models, Naive Bayes |
| **Performance** | Best results ~85–92% accuracy on structured activity datasets with SVM |
| **Limitations** | Requires physical sensor attachment; does not generalise to camera-based systems; limited to low-complexity activities |
| **Relation to FitVision** | Establishes the foundation that engineered hand-crafted features from body kinematics outperform raw signal inputs. FitVision follows this same principle — replacing raw pixel or coordinate data with computed joint distances — but shifts the sensor modality from wearables to camera-based pose estimation. |

### 2.3 Comparative Analysis of ML Classifiers for Exercise Recognition

**Reference:** Khurana, R., Srivastava, A., & Vatsa, M. (2021). *Pose-based exercise recognition using skeleton graph and gradient boosted trees.* Proceedings of the IEEE International Conference on Image Processing (ICIP), 2021.

| Aspect | Details |
| :--- | :--- |
| **Problem** | Classifying gym exercises (squat, lunge, deadlift, bicep curl) from skeleton pose sequences |
| **Dataset** | Custom video dataset with 300 clips across 6 exercise classes, processed with OpenPose |
| **Models** | Gradient Boosted Trees, Random Forest, and a Graph Convolutional Network (GCN) |
| **Performance** | GBT achieved 79.4% macro F1 on the held-out set; GCN reached 84.1% with temporal modelling |
| **Limitations** | Small dataset size; no handling of class imbalance; OpenPose is computationally expensive compared to MediaPipe |
| **Relation to FitVision** | Directly validates FitVision's model choices. The paper's finding that Gradient Boosted Trees outperform Random Forest on pose-based tabular features mirrors our XGBoost > RandomForest result. It also confirms that moving to a GCN with temporal modelling (future work for FitVision) can yield an additional ~5% improvement. |

**Cross-Paper Analysis:** All three works converge on a core insight: structured, engineered features from body landmarks — whether from wearables or vision-based pose estimators — consistently outperform raw feature inputs for activity classification. The progression from classical ML (Survey, 2014) → gradient-boosted methods (Khurana et al., 2021) → our work (FitVision, 2026) shows that the gap between classical and deep methods narrows significantly when domain-specific feature engineering is applied carefully.

---

## 3. Dataset Description

### 3.1 Dataset Source

FitVision uses a **custom-built dataset** assembled specifically for this project. Source videos were collected from publicly available exercise demonstration videos and curated to ensure consistent camera angle and lighting conditions. The raw videos were processed locally using the MediaPipe Pose framework to extract skeletal landmark data.

- **Dataset Type:** Tabular (post-feature-extraction from video)
- **Raw Input:** MP4 video files at varying frame rates, standardised to 10 FPS during processing
- **Processed Output:** `xyz_distances.csv` (features per frame) + `labels.csv` (video-level labels)

### 3.2 Dataset Characteristics

| Property | Value |
| :--- | :--- |
| **Exercise Classes** | Push-Up, Squat, Jumping Jack, Pull-Up |
| **Number of Videos** | ~400 total (varies per class) |
| **Number of Features** | 48 (engineered joint-distance features per frame) |
| **Feature Type** | Continuous numerical (3D Euclidean distances between joint pairs) |
| **Target Variable** | Exercise class label (multi-class: 4 categories) |
| **Frames per Video** | Variable (~50–200 frames at 10 FPS) |

### 3.3 Class Distribution

The dataset is **imbalanced across classes**:

| Class | Video Count | Approximate Frames |
| :--- | :---: | :---: |
| Jumping Jack | 107 | ~14,000 |
| Push-Up | ~90 | ~11,000 |
| Pull-Up | ~85 | ~10,000 |
| **Squat** | **63** | **~7,500** |

The Squat class is the minority class (~16% of total data), which directly impacts model performance and was a primary driver of our metric selection (Macro F1 over Accuracy).

### 3.4 Exploratory Data Analysis (EDA)

**Missing Value Analysis:** A rigorous NaN check confirmed **zero missing values** across all 48 feature columns after MediaPipe processing. Frames where MediaPipe failed to detect a pose were discarded during the data collection phase.

**Feature Distribution:** The 48 features represent Euclidean distances between anatomically meaningful joint pairs (e.g., `left_shoulder_left_wrist`, `left_hip_left_ankle`). These distances are strictly non-negative and exhibit right-skewed distributions due to body size variation across subjects.

**Inter-Class Separability:** Push-Up frames are geometrically distinct from all other classes because the body is horizontal (dominant X-Z plane motion). Jumping Jack and Pull-Up share vertical arm extension patterns, explaining their high confusion rate. Squats and Jumping Jacks share wide leg stances, also causing confusion.

**Correlation Structure:** Joint distances are naturally correlated (e.g., shoulder-to-wrist distance correlates with shoulder-to-elbow). This collinearity is handled implicitly by tree-based models but is a challenge for Logistic Regression.

### 3.5 Dataset Challenges and Limitations

| Challenge | Impact |
| :--- | :--- |
| **Class Imbalance** | Squat is underrepresented; accuracy metric is misleading without F1 |
| **Temporal Blindness** | Frames treated independently — temporal exercise dynamics are lost |
| **Body Size Variation** | Absolute distances vary across subjects; scaling is essential |
| **Camera Angle Variation** | Exercises filmed from different angles may yield inconsistent landmark projections |
| **Data Collection Bias** | Videos sourced from online demonstrations — may not reflect real-world gym conditions |

---

## 4. Data Preprocessing & Feature Engineering

### 4.1 Data Cleaning & Preprocessing

The initial dataset consisted of `xyz_distances.csv` containing distances and `labels.csv` containing video labels. We merged these on the `vid_id` to assign the correct exercise label to each individual frame. A rigorous `NaN` check confirmed zero missing values after MediaPipe extraction.

**Duplicate Removal:** Consecutive frames within the same video are near-identical. While not true duplicates, this temporal redundancy is managed by our video-level cross-validation strategy rather than explicit deduplication, preserving temporal density.

**Outlier Handling:** Extreme distance values can arise from MediaPipe landmark estimation errors (e.g., partially occluded limbs). StandardScaler's z-score normalisation implicitly reduces the influence of such outliers by centering and scaling each feature.

### 4.2 Scaling & Normalisation

To prevent **data leakage during scaling**, we strictly applied `StandardScaler` as follows:

```
scaler.fit_transform(X_train)  →  applied ONLY on training folds
scaler.transform(X_val/X_test) →  applied WITHOUT refitting on test data
```

This ensures that test-set statistics never influence the scaling parameters, preserving the integrity of the evaluation.

### 4.3 Feature Engineering

Rather than using raw coordinate inputs, our dataset inherently uses **engineered distance features**. We utilised MediaPipe to extract 33 pose landmarks, then explicitly calculated the 3D Euclidean distances (X, Y, Z components) between key joint pairs (e.g., `left_shoulder_left_wrist`, `left_hip_left_ankle`). We also created **average-based features** (distance from centre to midpoint of two joints). This resulted in **48 dense features per frame**, drastically improving model inputs compared to raw pixel data or absolute coordinates.

**Why distance features?** Absolute coordinates are camera-relative and change with subject position in the frame. Distances between joints are invariant to global translation, making them a more robust representation of body pose.

### 4.4 Train/Test Split & Reproducibility

**Splitting Strategy:** `StratifiedGroupKFold(n_splits=5)` with `random_state=42`, grouping by `vid_id`.

**Why not random split?** A naive `train_test_split` on frame rows yielded ~96% accuracy — this is **data leakage**. Adjacent frames from the *same video* ended up in both train and test sets, causing the model to memorise the subject/background rather than learning the exercise pattern. By grouping on `vid_id`, entire videos are isolated into separate folds, ensuring the model is evaluated on genuinely *unseen people and videos*.

**Preprocessing Pipeline Execution Order:**

```
Raw Videos
    ↓
MediaPipe Pose Extraction (33 landmarks × frame)
    ↓
Joint Distance Calculation (48 features per frame)
    ↓
Merge with Labels (vid_id join)
    ↓
NaN Check & Validation
    ↓
StratifiedGroupKFold Split (by vid_id)
    ↓
StandardScaler (fit on train, transform on val/test)
    ↓
Model Training & Evaluation
```

---

## 5. Model Selection & Justification

We evaluated three models across two distinct algorithmic families (Linear and Tree-based):

### Model 1 — Logistic Regression (Baseline)

| Property | Value |
| :--- | :--- |
| **Family** | Linear |
| **Solver** | `lbfgs` |
| **Max Iterations** | 1000 (ensures convergence) |
| **Multi-class Strategy** | One-vs-Rest (OvR) |

**Justification:** We started with a simple, interpretable linear model to establish a solid performance baseline before moving to complex algorithms. It provides a direct measure of linear separability in our 48 engineered features.

**Advantages:** Fast training, fully interpretable coefficients, no hyperparameter sensitivity.

**Limitations:** Cannot capture non-linear interactions between joint distances (e.g., a squat requires a specific *combination* of hip and knee angles, not just individual distances).

---

### Model 2 — Random Forest (Ensemble)

| Property | Value |
| :--- | :--- |
| **Family** | Tree-based (Bagging Ensemble) |
| **n_estimators** | 100 |
| **max_depth** | 10 |
| **random_state** | 42 |

**Justification:** Random Forest captures non-linear relationships and complex interactions between the joint distances, making it highly suitable for pose classification. The `max_depth=10` prevents overfitting on the highly correlated adjacent frames within training videos.

**Advantages:** Robust to outliers, handles feature correlations gracefully, provides feature importance rankings.

**Limitations:** Can still overfit on imbalanced classes; predictions are not as well-calibrated as boosted models.

---

### Model 3 — XGBoost (Gradient Boosted Trees)

| Property | Value |
| :--- | :--- |
| **Family** | Boosted Tree |
| **learning_rate** | 0.1 |
| **max_depth** | 6 |
| **n_estimators** | 100 |
| **random_state** | 42 |

**Justification:** Gradient boosting sequentially corrects errors from previous trees, representing the state-of-the-art for tabular data classification. Shallower trees (`max_depth=6`) are used because the boosting process itself handles complexity iteratively — this prevents overfitting while maintaining high expressiveness.

**Advantages:** Superior handling of imbalanced classes, built-in regularisation (L1/L2), consistently outperforms Random Forest on structured tabular data.

**Limitations:** More hyperparameters to tune; slower training than a single Random Forest for the same number of trees.

---

## 6. Evaluation Strategy & Metrics

### 6.1 Evaluation Setup

**Validation Strategy:** 5-Fold `StratifiedGroupKFold` cross-validation, grouped by `vid_id`.

**Why this setup?** Standard k-fold cross-validation on frame-level data causes data leakage (frames from the same video appear in both train and test folds). `StratifiedGroupKFold` solves this by:
1. **Grouping** — ensuring all frames from a given video stay in the same fold
2. **Stratifying** — maintaining class proportions across folds despite the grouping constraint

**Random Seed:** `42` used throughout for reproducibility.

**Leakage Prevention:** `StandardScaler` is fit exclusively on training folds and applied (without refitting) to validation folds. No test-set statistics were used at any point during training.

### 6.2 Evaluation Metrics

Because our dataset is **imbalanced** (Squat: 63 videos vs. Jumping Jack: 107 videos), accuracy alone is misleading. We report:

| Metric | Why It Matters |
| :--- | :--- |
| **Accuracy** | Overall correctness; reported for completeness but insufficient alone |
| **Macro Precision** | Average precision across all classes, equally weighted — penalises false positives on minority classes |
| **Macro Recall** | Average recall across all classes, equally weighted — penalises missed minority class samples |
| **Macro F1-Score** | Harmonic mean of macro precision and recall; our **primary metric** |
| **CV Macro F1** | 5-fold cross-validated macro F1; proves generalisation stability |

**Why Macro F1?** By averaging metrics *per class* (not per sample), Macro F1 ensures that the minority `squat` class receives equal weight to the majority `jumping_jack` class. A model that ignores squats entirely would score high on weighted metrics but would have a low Macro F1 — exactly the behaviour we want to penalise.

---

## 7. Results & Comparative Analysis

### 7.1 Model Performance Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | **CV Macro F1** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LogisticRegression (Baseline) | 67.34% | 66.37% | 64.77% | 64.65% | **65.52%** |
| RandomForest | 81.33% | 80.70% | 80.36% | 80.50% | **80.49%** |
| **XGBoost 🏆** | **82.30%** | **82.17%** | **81.32%** | **81.65%** | **82.37%** |

*All metrics reported on the held-out test fold of the final cross-validation split. CV Macro F1 averaged across all 5 folds.*

### 7.2 Visual Results

Confusion matrix images are included in the `/artifacts` directory:
- `cm_LogisticRegression_Baseline.png` — Logistic Regression confusion matrix
- `cm_RandomForest.png` — Random Forest confusion matrix
- `cm_XGBoost.png` — XGBoost confusion matrix (primary analysis)

### 7.3 Comparative Discussion

**Best Model — XGBoost:** XGBoost achieved the highest CV Macro F1 (**82.37%**), confirming its superiority for this tabular pose-classification task. Its sequential error-correction mechanism allowed it to better handle the class imbalance and non-linear joint interactions compared to both baselines.

**Baseline Comparison:** Logistic Regression's 65.52% CV Macro F1 establishes the upper bound of linear separability in our feature space. The 17-point gap between Logistic Regression and XGBoost (82.37% - 65.52%) confirms that non-linear feature interactions are essential for exercise classification.

**Overfitting/Underfitting:** The near-identical Macro F1 and CV Macro F1 values for XGBoost (81.65% vs. 82.37%) indicate no significant overfitting. The model generalises well to unseen videos.

**Computational Complexity:** Training time is negligible for all three models on this dataset size (~seconds). XGBoost's inference latency per frame is well within real-time requirements (<1ms per frame).

**Impact of Preprocessing:** The leakage-free evaluation reduced apparent accuracy from ~96% (naive random split) to ~82% (video-level split). This is not a performance regression — it is a correction from an artificially inflated baseline to an honest, scientific measurement.

**Trade-off Summary:** XGBoost is the recommended production model — it offers the best generalisation performance with negligible computational overhead. Random Forest is a reasonable alternative if interpretability is prioritised over raw performance.

---

## 8. Error Analysis & Critical Reflection

### 8.1 Confusion Matrix Analysis (XGBoost — Best Model)

Analysing the XGBoost confusion matrix reveals clear, explainable failure patterns:

**Class 1 — Squat (Hardest Class):**
Squats have the lowest per-class performance. XGBoost misclassified 74 squat frames as jumping jacks, and 36 as pull-ups.
- **Root Cause:** Squats have the fewest training samples (only 63 videos). Furthermore, the lower-body mechanics of a deep squat — wide knee separation, lowered hip position — can mathematically resemble the wide-legged stance of a jumping jack mid-jump when viewed as a single static frame.

**Class 2 — Push-Up (Easiest Class):**
Push-ups were classified near-perfectly.
- **Root Cause:** Push-ups are geometrically unique — the entire body is horizontal, making the X-Z plane dominant and the Y-axis (vertical) compressed. No other exercise in the dataset shares this spatial signature.

**Class 3 — Pull-Up vs. Jumping Jack Confusion:**
68 pull-up frames were misclassified as jumping jacks, and 83 jumping jack frames were misclassified as pull-ups.
- **Root Cause:** Both exercises feature vertical arm extensions above the head. Because our model analyses each frame independently (no temporal context), the apex of a jumping jack's arm extension is geometrically indistinguishable from the apex of a pull-up in terms of static joint distances. This is a fundamental limitation of our frame-independent approach.

### 8.2 Project Limitations

| Limitation | Severity | Notes |
| :--- | :---: | :--- |
| **Class Imbalance** | High | Squat underrepresentation degrades minority-class recall |
| **No Temporal Modelling** | High | Each frame is treated independently; exercise dynamics are ignored |
| **Data Leakage (addressed)** | Resolved | Switching to video-level CV reduced F1 from 96% to 82% — honest metric |
| **Limited Exercise Vocabulary** | Medium | Only 4 exercise types; real-world fitness applications need 20+ |
| **Camera Angle Sensitivity** | Medium | MediaPipe accuracy degrades with non-frontal camera angles |

### 8.3 Honest Reflection on Results

Our most important finding is the **data leakage discovery**: by switching from naive frame-level `train_test_split` to `StratifiedGroupKFold` (video-level splitting), our Macro F1 dropped from ~96% to 82%. While 82% seems lower, it is a **scientifically honest metric** — it proves the model can generalise to videos it has never seen, which is the only performance that matters in production deployment.

---

## 9. Conclusion & Future Work

### 9.1 Summary of Findings

FitVision successfully built a complete ML pipeline for exercise classification using MediaPipe pose estimation and 48 engineered joint-distance features. Three models were trained and evaluated using a rigorous, leakage-free cross-validation strategy:

- **XGBoost** is the production-ready model with a CV Macro F1 of **82.37%**
- **Random Forest** is a strong alternative at 80.49% CV Macro F1
- **Logistic Regression** established a linear baseline at 65.52%, confirming the necessity of non-linear models

The most significant technical contribution of this project is the identification and resolution of a data leakage problem that inflated naive results from 96% to an honest 82%.

### 9.2 Key Lessons Learned

1. **Data leakage is subtle but critical:** Frame-level splitting with video datasets creates a deceptive performance illusion. Always split at the natural unit of independence (the video/subject level).
2. **Feature engineering beats raw data:** 48 distance features from 33 landmarks proved more informative than using raw coordinates — exercise classification is about body geometry, not pixel values.
3. **Macro F1 is the right metric for imbalanced multi-class problems:** Accuracy would have hidden the model's weakness on the minority squat class.

### 9.3 Future Work

| Improvement | Expected Impact |
| :--- | :--- |
| **Temporal Modelling (LSTM / 1D-CNN)** | Address the Pull-Up vs. Jumping Jack confusion by learning motion sequences |
| **SMOTE / Class Weighting** | Improve minority class (Squat) recall by addressing training imbalance |
| **Hyperparameter Tuning (GridSearch/Optuna)** | Potential 2–5% improvement in Macro F1 |
| **Expanded Exercise Vocabulary** | Scale from 4 to 20+ exercise classes for real-world applicability |
| **On-device Inference Optimisation** | Quantise models for mobile deployment alongside MediaPipe |
| **Real-time Form Feedback** | Extend classification to per-rep quality scoring (e.g., squat depth metric) |

---

## 10. Teamwork & Task Delegation

| Team Member Name | Student ID | Assigned Role | Main Responsibilities | Deliverables Contributed |
| :--- | :--- | :--- | :--- | :--- |
| Omar Hesham | 247469 | ML Lead / Coordinator | Model 1 (Logistic Regression) implementation, training runs, experiment tracking, and workflow coordination | Model 1 training, evaluation, and comparative analysis |
| Mahmoud Abdelkareem | 247519 | Data Engineer | Dataset preprocessing, feature engineering pipeline, StandardScaler implementation, and leakage-free split design | Dataset description section and preprocessing pipeline code |
| Ahmed Ebrahim | 246057 | Evaluation & Metrics Lead / Error Analyst | Evaluation strategy design, metric selection, StratifiedGroupKFold setup, and confusion matrix error analysis | Evaluation strategy section and error analysis section |
| Ahmed Salah | 247841 | ML Engineer | Model 2 (Random Forest) implementation, hyperparameter selection, training, and result tracking | Random Forest model implementation and comparative results |
| Noureen Mohamed | 246173 | ML Engineer | Model 3 (XGBoost) implementation, hyperparameter selection, training, and result tracking | XGBoost model implementation and comparative results |

*Each member contributed to evaluation, data validation, and results discussion across all sections. Roles define primary responsibility, not exclusive ownership.*

---

## Appendix

### A. Confusion Matrix Artefacts

All confusion matrices are stored as both `.png` (visual) and `.txt` (raw values) files in the `artifacts/` directory:

```
artifacts/
├── cm_LogisticRegression_Baseline.png
├── cm_LogisticRegression_Baseline.txt
├── cm_RandomForest.png
├── cm_RandomForest.txt
├── cm_XGBoost.png
├── cm_XGBoost.txt
├── report_LogisticRegression_(Baseline).txt
├── report_RandomForest.txt
└── report_XGBoost.txt
```

### B. Reproducibility

All experiments can be reproduced by running:

```bash
cd FitVision/app/ml/training
python train_models_comparison.py
```

**Requirements:** `random_state=42` is set globally in all models and splitting functions. Full dependency list in `requirements.txt`.

### C. Model Artefacts

Trained model files saved in `app/ml/models/`:
- `workout_classifier.pkl` — Production XGBoost model
- `scaler.pkl` — Fitted StandardScaler
- `label_encoder.pkl` — Label-to-integer encoder
- `feature_columns.pkl` — Ordered feature column names

---

## References

1. Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020). *BlazePose: On-device Real-time Body Pose tracking.* CVPR Workshop on Computer Vision for Augmented and Virtual Reality, Google Research.

2. Bulling, A., Blanke, U., & Schiele, B. (2014). *A tutorial on human activity recognition using body-worn inertial sensors.* ACM Computing Surveys, 46(3), 33:1–33:33.

3. Khurana, R., Srivastava, A., & Vatsa, M. (2021). *Pose-based exercise recognition using skeleton graph and gradient boosted trees.* Proceedings of the IEEE International Conference on Image Processing (ICIP), 2021, pp. 2908–2912.
