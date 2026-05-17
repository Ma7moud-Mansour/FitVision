# Machine Learning (CS 363) — Course Project
## Milestone 2: Model Training & Evaluation Report
**Project Name:** FitVision
**Team:** 5 Members

---

### Team Contribution Table

| Team Member Name | Student ID | Assigned Role | Main Responsibilities | Deliverables Contributed |
| :--- | :--- | :--- | :--- | :--- |
| *[Mahmoud Abdelkareem]* | *[247519]* | ML Lead / Coordinator | Model 1 selection, training runs, and tracking | Model training & results section |
| *[Omar Hesham]* | *[247604]* | Data Engineer | Preprocessing pipeline, feature engineering | Data section & pipeline code |
| *[Ahmed Ebrahim]* | *[ID 3]* | Eval & Metrics Lead | Evaluation strategy, metric selection, cross-validation | Evaluation & Error analysis |
| *[Ahmed Salah]* | *[ID 4]* | ML Engineer | Model 2 selection, training runs | Model training & results section |
| *[Noureen Mohamed]* | *[ID 5]* | ML Engineer | Model 3 selection, training runs | Model training & results section |

---

### 1. Data Preprocessing & Feature Engineering

**Data Cleaning & Preprocessing:**
The initial dataset consisted of `xyz_distances.csv` containing distances and `labels.csv` containing video labels. We merged these on the `vid_id` to assign the correct exercise label to each individual frame. A rigorous `NaN` check confirmed zero missing values. To prevent any data leakage during scaling, we initialized a `StandardScaler` and applied `fit_transform()` exclusively to the training set, while only applying `transform()` to the validation/test set.

**Feature Engineering:**
Rather than using raw coordinate inputs, our dataset inherently uses engineered features. We utilized MediaPipe to extract 33 pose landmarks, but then explicitly calculated the 3D distances (X, Y, Z) between key joints (e.g., `left_shoulder_left_wrist`, `left_hip_left_ankle`). We also created average-based features (distance from center to midpoint of two joints). This resulted in 48 dense features per frame, drastically improving the quality of model inputs compared to raw pixel data or absolute coordinates.

---

### 2. Model Selection & Justification

We evaluated three models across two distinct families (Linear and Tree-based):

1. **Logistic Regression (Baseline):** 
   * **Family:** Linear.
   * **Justification:** We started with a simple, interpretable linear model to establish a solid performance baseline before moving to complex algorithms. It provides a measure of linear separability in our engineered features.
   * **Hyperparameters:** `solver="lbfgs"`, `max_iter=1000` (to ensure convergence).

2. **RandomForestClassifier:**
   * **Family:** Tree-based (Ensemble).
   * **Justification:** Random Forest captures non-linear relationships and complex interactions between the joint distances, making it highly suitable for pose classification.
   * **Hyperparameters:** `n_estimators=100` (good balance between performance and compute), `max_depth=10` (preventing over-fitting on highly correlated adjacent frames).

3. **XGBoost Classifier:**
   * **Family:** Boosted Tree.
   * **Justification:** Gradient boosting sequentially corrects errors from previous trees, often representing the state-of-the-art for tabular data. We hypothesized it would outperform RandomForest.
   * **Hyperparameters:** `learning_rate=0.1`, `max_depth=6` (shallower trees because boosting handles complexity iteratively), `n_estimators=100`.

---

### 3. Evaluation Strategy & Metrics

**Data Leakage Prevention (Crucial Finding):**
Initially, using a simple random `train_test_split` on frames yielded an artificially high accuracy (~96%). However, this causes severe **data leakage** because adjacent frames from the *same video* end up in both the training and test sets. The model was learning the specific person/background rather than generalizing the exercise. 
**Solution:** We replaced the split with `StratifiedGroupKFold(n_splits=5)`, explicitly grouping by `vid_id`. This isolates entire videos into either the train or test set, proving the model can generalize to *unseen people/videos*. 

**Metrics:**
Because our dataset is imbalanced (e.g., 107 `jumping_jack` videos vs. only 63 `squat` videos), accuracy alone is misleading. We rely primarily on **Macro F1-Score** to ensure that minority classes like `squat` are weighted equally to majority classes. We also report 5-fold Cross-Validated Macro F1 to prove generalization stability.

---

### 4. Results & Comparative Analysis

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | **CV Macro F1** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LogisticRegression (Baseline) | 67.34% | 66.37% | 64.77% | 64.65% | **65.52%** |
| RandomForest | 81.33% | 80.70% | 80.36% | 80.50% | **80.49%** |
| **XGBoost 🏆** | **82.30%** | **82.17%** | **81.32%** | **81.65%** | **82.37%** |

**Conclusion:** XGBoost is the clear winner, achieving a robust Cross-Validated Macro F1 of **82.37%**. It successfully captured non-linear joint interactions while maintaining generalization across unseen videos.

---

### 5. Error Analysis

Analyzing the XGBoost Confusion Matrix reveals clear failure patterns:

1. **The Squat Problem (Hardest Class):**
   * Squats have the lowest performance. XGBoost misclassified 74 squat frames as jumping jacks, and 36 as pull-ups.
   * **Why?** Squats have the fewest training samples (only 63 videos). Furthermore, the lower-body mechanics of a deep squat can mathematically resemble the wide-legged stance of a jumping jack. 

2. **The Push-Up Success (Easiest Class):**
   * Push-ups are perfectly distinct from all other exercises because the entire body is horizontal (X-Z plane dominant vs. Y-axis dominant). XGBoost achieved near-perfect classification here.

3. **Pull-up vs. Jumping Jack Confusion:**
   * 68 pull-up frames were predicted as jumping jacks, and 83 jumping jack frames as pull-ups.
   * **Why?** Both exercises feature vertical arm extensions above the head. Because our model looks at static frames independently (no temporal awareness), the apex of a jumping jack looks identical to the apex of a pull-up in terms of arm coordinates.

---

### 6. Critical Reflection & Limitations

Our audit of the model reveals three primary limitations:

1. **The Reality of Data Leakage:** By switching to `StratifiedGroupKFold` (video-level splitting), our Macro F1 dropped from 96% to 82%. While 82% seems lower, it is a scientifically honest metric. We successfully eliminated the data leakage that was artificially inflating our results.
2. **Class Imbalance:** The model still struggles with minority classes (`squat`). Future work should explore SMOTE (Synthetic Minority Over-sampling Technique) or class weighting in the loss function to penalize misclassifications of minority classes.
3. **Lack of Temporal Modeling:** Our pipeline treats every single frame as an independent image. Exercises are temporal sequences, not static poses. The confusion between a jumping jack and a pull-up proves that relying solely on static geometry is a limitation. Moving to an LSTM or 1D-CNN over the frame sequence would represent a massive leap in accuracy.
