# AIAA3111 Group Project

## Project title

**Context-Aware Anomaly Detection in NASA Spacecraft Telemetry Data**

## 1. Project type

This is a data mining project in the **anomaly detection** category.

The task is to learn the normal behavior of multivariate spacecraft telemetry and identify time points or time intervals that deviate from the learned pattern.

\[
X_{1:T}\rightarrow \hat{y}_{1:T},\qquad y_t\in\{0,1\}
\]

where $y_t=1$ means that the telemetry at time $t$ is anomalous.

The provided anomaly intervals are used as ground truth for evaluation. The main models can be trained using normal training data, so the project remains an anomaly detection study rather than a simple supervised classification exercise.

## 2. Motivation

Spacecraft telemetry contains measurements from many sensors and subsystems. A fault may appear as an unusual value in one channel, a change in temporal behavior, or an inconsistent pattern across several channels. Detecting these events early is important for reliable spacecraft operation.

The project studies whether using temporal and multivariate context improves anomaly detection compared with methods that inspect each signal independently.

## 3. Dataset

We will use the NASA SMAP and MSL spacecraft telemetry datasets.

- **SMAP**: Soil Moisture Active Passive satellite telemetry.
- **MSL**: Mars Science Laboratory telemetry.
- Each dataset contains multivariate time-series channels.
- The dataset provides normal training data, test data, and annotated anomaly intervals.
- The interval annotations will be converted into point-wise labels for evaluation.

\[
y_t=1\quad\text{if }t\text{ belongs to an annotated anomaly interval};
\qquad y_t=0\quad\text{otherwise}
\]

We will first use one dataset for the main experiments and use the other as an external test set if the preprocessing is compatible.

## 4. Research questions

1. Can a model trained on normal telemetry detect the labeled anomalies?
2. Does multivariate context improve over independent univariate detection?
3. Which model gives the best trade-off between detection quality and detection delay?
4. Do different anomaly types produce different error patterns?

## 5. Methods

### 5.1 Preprocessing

1. Load the train, test, channel, and anomaly-interval files.
2. Check missing values, sequence lengths, channel dimensions, and timestamp order.
3. Normalize each channel using statistics from the training set only.
4. Split the training sequence into fixed-length windows.
5. Convert anomaly intervals into point-wise and window-level labels.

For a window $W_i$, the window label can be defined as:

\[
y_i^{window}=1\quad\text{if }\sum_{t\in W_i}y_t>0
\]

### 5.2 Baselines

- Global or rolling z-score threshold.
- PCA reconstruction error.
- Isolation Forest.
- One-Class SVM, if the computational cost is acceptable.

### 5.3 Main models

- LSTM Autoencoder or GRU Autoencoder.
- A temporal prediction model whose residual is used as an anomaly score.
- Optional: Transformer encoder if the baseline experiments are stable.

For reconstruction-based detection:

\[
e_t=\|x_t-\hat{x}_t\|_2
\]

For prediction-based detection:

\[
e_t=\|x_t-\hat{x}_t\|_2
\]

A threshold $\tau$ is selected on a validation set:

\[
\hat{y}_t=\mathbb{1}(e_t>\tau)
\]

### 5.4 Context ablation

We will compare at least three input settings:

1. **Univariate**: one channel at a time;
2. **Multivariate**: all available channels jointly;
3. **Temporal context**: a window of previous observations.

If the data organization permits, an additional channel-group experiment will compare measurements from the same subsystem with measurements from unrelated channels.

## 6. Evaluation

The main metrics are:

\[
\text{Precision}=\frac{TP}{TP+FP}
\]

\[
\text{Recall}=\frac{TP}{TP+FN}
\]

\[
F1=\frac{2\cdot\text{Precision}\cdot\text{Recall}}
{\text{Precision}+\text{Recall}}
\]

\[
\text{FPR}=\frac{FP}{FP+TN}
\]

For each labeled anomaly interval $[t_s,t_e]$, detection delay is:

\[
\text{Delay}=\hat{t}_{first}-t_s
\]

where $\hat{t}_{first}$ is the first detected anomalous time point after $t_s$. We will report both point-wise metrics and event-level detection rate, because a model that detects one point in an interval should not be treated exactly the same as a model that detects the whole interval.

## 7. Expected contribution

The project will provide a reproducible comparison of classical and neural anomaly detection methods on NASA spacecraft telemetry. The main conclusion will be based on whether temporal and multivariate context improves F1, false-positive rate, and detection delay over simple baselines.

## 8. Implementation plan

### Phase 1: Dataset and labels

- Download and inspect SMAP/MSL.
- Write a loader and label-conversion script.
- Produce a data summary table and sample plots.
- Select one main dataset and freeze the train/validation/test split.

### Phase 2: Baselines

- Implement z-score, PCA, and Isolation Forest.
- Produce the first metric table.
- Verify that the evaluation code handles anomaly intervals correctly.

### Phase 3: Neural models

- Implement the LSTM/GRU Autoencoder.
- Tune window length, hidden dimension, and threshold on validation data only.
- Run the same test protocol as the baselines.

### Phase 4: Analysis

- Run univariate versus multivariate ablations.
- Analyze false positives and missed anomaly intervals.
- Measure detection delay and runtime.
- Repeat the strongest method on the second dataset if feasible.

### Phase 5: Deliverables

- Freeze code and environment instructions.
- Write the report and README.
- Create the poster with the problem, data, methods, results, and limitations.
- Prepare a short presentation and anticipated questions.

## 9. Division of work

- **Data and labels**: dataset download, parsing, normalization, interval conversion.
- **Baselines**: statistical methods, PCA, Isolation Forest, evaluation scripts.
- **Neural model**: LSTM/GRU Autoencoder and hyperparameter experiments.
- **Analysis and presentation**: plots, tables, report, poster, and README integration.

Everyone should review the final code and report so that the submitted work is reproducible as a group.

## 10. Main risks and controls

- **Anomaly imbalance**: report precision, recall, F1, FPR, and event-level metrics rather than accuracy alone.
- **Threshold leakage**: select thresholds using validation data only.
- **Channel scale differences**: fit normalization on training data only.
- **Interval-label ambiguity**: report both point-wise and event-level results.
- **Overfitting one dataset**: keep the preprocessing and evaluation protocol fixed, and use the second dataset as an external check when possible.

## Basic proposal draft

**Project Title:** Context-Aware Anomaly Detection in NASA Spacecraft Telemetry Data

**Brief Project Description:**

This project investigates multivariate time-series anomaly detection using the NASA SMAP and MSL spacecraft telemetry datasets. The datasets contain measurements from multiple spacecraft channels and annotated anomaly intervals. We will compare statistical and machine-learning baselines, including z-score thresholds, PCA reconstruction error, Isolation Forest, and an LSTM or GRU autoencoder. The models will be trained mainly on normal telemetry and evaluated against the provided anomaly labels. We will study whether temporal windows and multivariate context improve anomaly detection performance compared with independent channel-based detection. Performance will be measured using precision, recall, F1-score, false positive rate, event-level detection rate, and detection delay. The final submission will include reproducible code, a README, the report, and instructions for downloading the dataset.

**Project Type:** Anomaly detection

**Planned Datasets:** NASA SMAP and MSL spacecraft telemetry datasets

**Expected Outputs:** Baseline comparison, context ablation study, error analysis, report, poster, code, and README.
