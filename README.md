# Representation-Dependent Learning Behaviour in Machine Learning-Based Intrusion Detection

Reproducibility repository for the study:

> **Representation-Dependent Learning Behaviour in Machine Learning-Based Intrusion Detection: Experimental Evidence and the Representation-Dependent Discriminative Structure Principle**

The repository contains the experimental configuration, representation definitions, preprocessing procedures, model implementations, evaluation procedures, calibration analysis, statistical analysis, sensitivity analysis, and table-generation utilities associated with the manuscript.

---

## 1. Study scope

The experimental design evaluates the interaction among:

- **Datasets**
  - CIC-DDoS2019
  - CICIoT2023
  - CIC-IoT-DIAD2024

- **Representations**
  - Flow
  - Header
  - Hybrid

- **Architectures**
  - Logistic Regression (LR)
  - Random Forest (RF)
  - XGBoost (XGB)
  - 1D CNN
  - BiLSTM

The principal experimental structure is:

```text
Dataset × Representation × Architecture × Run
```

The primary repeated seeds are:

```text
2026, 2027, 2028, 2029, 2030,
2031, 2032, 2033, 2034, 2035
```

The controlled classification setting uses the three target classes:

```text
Benign
SYN_Flood
UDP_Flood
```

The manuscript's broader scope limitations should be retained when interpreting the results. The controlled evaluation does not by itself establish general IDS performance across attack families outside the evaluated class space.

---

## 2. Repository structure

```text
representation-dependent-ids/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   ├── CIC-DDoS2019/
│   │   ├── Flow/
│   │   ├── Header/
│   │   └── Hybrid/
│   ├── CICIoT2023/
│   │   ├── Flow/
│   │   ├── Header/
│   │   └── Hybrid/
│   └── CIC-IoT-DIAD2024/
│       ├── Flow/
│       ├── Header/
│       └── Hybrid/
│
├── configuration/
│   ├── experiment_configuration.yaml
│   ├── seeds.txt
│   └── feature_configuration.yaml
│
├── code/
│   ├── preprocessing.py
│   ├── representation_builder.py
│   ├── classical_models.py
│   ├── deep_models.py
│   ├── evaluation.py
│   ├── calibration.py
│   ├── run_classical.py
│   ├── run_deep_learning.py
│   └── reliability_diagram.py
│
├── analysis/
│   ├── statistical_analysis.py
│   ├── rsi_analysis.py
│   ├── sensitivity_analysis.py
│   └── generate_tables.py
│
├── figures/
│   └── figure8_reliability.py
│
├── feature_mapping/
│   ├── CIC-DDoS2019.csv
│   ├── CICIoT2023.csv
│   └── CIC-IoT-DIAD2024.csv
│
└── results/
    ├── README.md
    └── .gitkeep
```

Some directories contain placeholders until the corresponding released data or analysis outputs are populated.

---

## 3. Reproducibility status

This repository has been organized from:

1. the experimental procedures documented in the manuscript;
2. the available source implementations used during the study;
3. the documented model configurations and evaluation definitions; and
4. the revised reproducibility requirements adopted during manuscript revision.

It is therefore a **clean reproducibility repository**, rather than a verbatim archive of every development-stage file used during the research process.

Development notebooks, exploratory scripts, temporary files, intermediate exports, and obsolete experiment versions are intentionally not included unless they are required to reproduce the documented analysis.

Where a historical source implementation was available, the corresponding repository module preserves its documented configuration. Where only the manuscript and available experimental information support a procedure, the repository implements that documented procedure without claiming that the reconstructed script is the original development file.

---

## 4. Data

The original benchmark datasets are not redistributed in this repository.

The benchmark datasets used by the study are:

- CIC-DDoS2019
- CICIoT2023
- CIC-IoT-DIAD2024

The repository is intended to contain the **derived representation-level files** required by the experiments:

```text
Flow
Header
Hybrid
```

with separate:

```text
train.csv
validation.csv
test.csv
```

partitions.

The released derived data are generated from the benchmark datasets under the controlled preprocessing and harmonization procedures described in the manuscript.

The original benchmark datasets remain subject to their respective source terms and are not redistributed here.

---

## 5. Representation definitions

The representation space is explicitly separated from the dataset identity.

### Flow representation

The documented CICIoT2023 Flow representation contains:

```text
Rate
Tot sum
Min
Max
AVG
Std
Tot size
IAT
Number
Variance
```

### Header representation

The documented CICIoT2023 Header representation contains the header/protocol attributes defined by the available representation-construction source.

### Hybrid representation

The Hybrid representation combines the Header and Flow feature groups.

The repository does not assume that a dataset is itself a representation. Dataset and representation are maintained as separate experimental factors throughout the analysis.

For datasets where a complete feature mapping is not available in the released construction source, the repository does not fabricate a mapping.

---

## 6. Sampling and partitioning

The available CICIoT2023 representation-construction procedure documents:

- 56,863 samples per target class;
- classes:
  - BenignTraffic
  - SYN_Flood
  - UDP_Flood;
- chunked processing for memory management;
- random seed 2026 for the representation-construction sampling procedure;
- class-wise 70:10:20 train/validation/test partitioning;
- group-aware handling of identical feature vectors;
- cross-partition duplicate checking.

The resulting target is a class-balanced dataset with:

```text
56,863 × 3 = 170,589 samples
```

when the complete balanced construction is used.

Other dataset-specific construction details are retained only where supported by the available source implementation or manuscript documentation.

---

## 7. Preprocessing

Data-dependent preprocessing parameters are fitted using the training partition only.

The documented procedure includes:

1. identify the label column;
2. exclude label and source-file metadata;
3. convert feature values to numeric form;
4. replace infinite values with missing values;
5. fit median imputation on the training data;
6. fit Min-Max scaling on the training data;
7. apply the fitted transformations unchanged to validation and test data.

The independent test partition is not used to fit preprocessing parameters.

For classical models, the preprocessing is implemented inside the model pipeline so that fitting occurs as part of training.

For deep-learning experiments, the repository provides an explicit training-fitted preprocessing path.

---

## 8. Classical machine-learning models

The documented primary model configurations are:

### Logistic Regression

```text
Penalty: L2
Solver: lbfgs
Maximum iterations: 1000
```

Preprocessing:

```text
Median imputation → Min-Max scaling → Logistic Regression
```

### Random Forest

```text
n_estimators = 200
max_depth = 20
criterion = gini
```

### XGBoost

```text
n_estimators = 300
learning_rate = 0.05
max_depth = 10
subsample = 0.8
objective = multi:softprob
eval_metric = mlogloss
```

Random state is set from the experimental seed.

The available earlier `04_run_pilot.py` is explicitly treated as a **pilot implementation**. The repository's `run_classical.py` provides the documented Dataset × Representation × Architecture × Seed execution structure and should not be described as proof that the pilot script historically generated every final manuscript result.

---

## 9. Deep-learning models

### 1D CNN

The available full experiment implementation uses:

```text
Conv1D
    filters = 64
    kernel_size = 3
    activation = ReLU
    padding = same

Dropout = 0.3

GlobalMaxPooling1D

Dense
    activation = softmax
```

### BiLSTM

The available full experiment implementation uses two bidirectional LSTM layers:

```text
Bidirectional LSTM
    units = 128
    dropout = 0.3
    return_sequences = True

Bidirectional LSTM
    units = 128
    dropout = 0.3
    return_sequences = False

Dense
    activation = softmax
```

### Training configuration

```text
Optimizer: Adam
Learning rate: 0.001
Loss: categorical cross-entropy
Batch size: 128
Maximum epochs: 50
Early stopping monitor: validation loss
Early stopping patience: 8
Restore best weights: True
```

The deep-learning input is reshaped to:

```text
(samples, features, 1)
```

for the sequence/convolutional architectures.

---

## 10. Evaluation metrics

The repository evaluates the multiclass task using:

- Accuracy
- Per-class Precision
- Per-class Recall
- Macro Precision
- Macro Recall
- Macro-F1
- Weighted F1
- Balanced Accuracy
- Multiclass Matthews Correlation Coefficient (MCC)
- Multiclass Brier Score
- Expected Calibration Error (ECE)

Macro-F1 is used as the principal performance measure because it gives equal consideration to the three target classes.

MCC is included as a complementary multiclass measure.

---

## 11. Brier Score

For the three-class setting, the multiclass Brier Score is calculated as:

\[
BS =
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{c=1}^{3}
(p_{ic}-y_{ic})^2
\]

where:

- \(N\) is the number of test observations;
- \(p_{ic}\) is the predicted probability for class \(c\);
- \(y_{ic}\) is the one-hot encoded true class indicator.

The implementation is provided in:

```text
code/evaluation.py
```

---

## 12. Expected Calibration Error

ECE uses the top-label calibration definition documented for the revised manuscript.

For each test observation:

```text
confidence = maximum predicted class probability
correctness = predicted class == true class
```

The confidence interval is divided into:

```text
10 equally spaced bins
```

For non-empty bin \(b\):

\[
ECE =
\sum_b
\frac{n_b}{N}
\left|
Acc_b-Conf_b
\right|
\]

where:

- \(n_b\) is the number of observations in bin \(b\);
- \(Acc_b\) is empirical accuracy;
- \(Conf_b\) is mean predicted confidence.

The ECE implementation is in:

```text
code/calibration.py
```

The calculation is top-label ECE, not a classwise ECE.

---

## 13. Reliability diagrams

The revised Figure 8 uses conventional reliability diagrams.

Scope:

```text
Dataset:       CIC-DDoS2019
Representations:
               Flow
               Header

Architectures:
               LR
               RF
               XGB
               CNN
               BiLSTM
```

The diagrams use:

```text
X-axis: Mean predicted confidence
Y-axis: Empirical accuracy
Bins:   10 equally spaced confidence bins
```

The diagonal reference line:

```text
y = x
```

represents perfect calibration.

No categorical calibration labels are assigned.

The numerical calibration results remain in the Brier Score and ECE tables.

---

## 14. Representation Stability Index (RSI)

RSI is implemented as a **domain-specific descriptive dispersion measure**.

For representation-level values:

\[
x_1,x_2,x_3
\]

corresponding to:

```text
Flow
Header
Hybrid
```

the index is:

\[
RSI =
\frac{SD(x_1,x_2,x_3)}
{\operatorname{Mean}(x_1,x_2,x_3)}
\]

using the sample standard deviation.

The repository does not assign:

- predefined RSI thresholds;
- stability categories;
- "high/moderate/low" labels;
- universal stability interpretations.

For the manuscript-style summary, dataset-level RSI values are summarized across the three benchmark datasets as:

```text
mean ± SD
```

The implementation is in:

```text
analysis/rsi_analysis.py
```

---

## 15. Statistical analysis

The primary statistical analysis operates on run-level observations.

The documented factorial structure is:

```text
Dataset
Representation
Architecture
Run
```

The principal interaction of interest is:

```text
Representation × Architecture
```

The full factorial model additionally evaluates:

```text
Dataset × Representation × Architecture
```

The implementation supports:

- Type III ANOVA;
- sum-to-zero contrasts;
- partial eta-squared;
- descriptive mean ± SD;
- 95% confidence intervals;
- dataset-specific Representation × Architecture analyses.

The implementation is in:

```text
analysis/statistical_analysis.py
```

Statistical analyses should be run from run-level results rather than from already averaged manuscript tables.

---

## 16. Sensitivity and robustness analysis

The repository provides two documented sensitivity/audit procedures.

### Feature-order sensitivity

Feature order can be permuted while preserving:

- the same feature values;
- the same feature set;
- the same labels;
- the same train/validation/test observations.

The repository records a deterministic feature-order hash for each permutation.

Performance changes can be summarized relative to the identity ordering.

No arbitrary acceptable-change threshold is imposed by the repository.

### Feature-level predictive-concentration audit

Training-only feature association measures include:

- Mutual Information
- ANOVA F-statistic

These are audit measures for examining feature-level predictive concentration.

They are not interpreted as causal effects and are not, by themselves, proof of data leakage.

The implementation is in:

```text
analysis/sensitivity_analysis.py
```

---

## 17. Table generation

The central table-generation script is:

```text
analysis/generate_tables.py
```

It generates the manuscript-oriented tables from a common representation-level intermediate dataset.

### Table 10

Dataset-level classification performance.

For each:

```text
Dataset × Architecture
```

the value is:

\[
\frac{
Flow + Header + Hybrid
}{3}
\]

using the corresponding native-dimensionality representation-level values.

### Table 11

Representation-level native-dimensionality results:

```text
Dataset × Representation × Architecture
```

### Table 13

Cross-representation characteristics, including:

- Flow
- Header
- Hybrid
- highest representation
- lowest representation
- range

No categorical stability label is assigned.

### Table 14

Aggregate Macro-F1 separately for:

```text
Flow
Header
Hybrid
```

across the three datasets, together with highest/lowest representation and range.

### Table 15

RSI across Flow/Header/Hybrid and aggregation of dataset-level RSI values as mean ± SD.

### Table 16

Dataset × Architecture Brier Score:

\[
\operatorname{Mean}
(Brier_{Flow},Brier_{Header},Brier_{Hybrid})
\]

### Table 17

Dataset × Architecture ECE using the same arithmetic representation-level aggregation.

---

## 18. Running the primary experiments

After the derived representation files have been placed under `data/` and the required dependencies have been installed:

### Classical models

```bash
python code/run_classical.py \
    --data-root data \
    --output results/classical_primary_results.csv
```

The runner is designed for:

```text
3 datasets
× 3 representations
× 3 classical architectures
× 10 seeds
= 270 experimental conditions
```

### Deep learning

```bash
python code/run_deep_learning.py \
    --data-root data \
    --output results/deep_learning_primary_results.csv
```

The documented primary deep-learning structure is:

```text
3 datasets
× 3 representations
× 2 architectures
× 10 seeds
= 180 experimental conditions
```

Probability outputs can additionally be saved for calibration/reliability analysis.

---

## 19. Running statistical analysis

Example:

```bash
python analysis/statistical_analysis.py \
    --results results/classical_primary_results.csv \
    --output-dir results/statistical_analysis
```

For the requested metrics:

```bash
python analysis/statistical_analysis.py \
    --results results/classical_primary_results.csv \
    --metrics Macro_F1,MCC
```

---

## 20. Running RSI analysis

Example:

```bash
python analysis/rsi_analysis.py \
    --results results/classical_primary_results.csv \
    --output-dir results/rsi_analysis
```

The default metrics are:

```text
Macro_F1
MCC
```

---

## 21. Running sensitivity analysis

Generate a deterministic feature-order manifest:

```bash
python analysis/sensitivity_analysis.py manifest \
    --features feature_mapping/CICIoT2023_features.txt \
    --n-permutations 10 \
    --seed 2026 \
    --output results/sensitivity/feature_order_manifest.csv
```

Analyse completed permutation results:

```bash
python analysis/sensitivity_analysis.py compare \
    --results results/sensitivity/feature_order_results.csv \
    --metric Macro_F1 \
    --output-dir results/sensitivity/feature_order
```

The feature-level predictive-concentration functions can also be imported directly from the module for training-only audits.

---

## 22. Generating manuscript tables

After the primary run-level result file has been produced:

```bash
python analysis/generate_tables.py \
    --results results/classical_primary_results.csv \
    --output-dir results/tables
```

The generated files include:

```text
representation_level_means.csv

table10_dataset_level_performance.csv
table11_representation_level_results.csv
table13_stability_characteristics.csv
table14_aggregate_macro_f1.csv
table15_rsi.csv
table15_rsi_dataset_components.csv
table16_brier_score.csv
table17_ece.csv
```

The intermediate `representation_level_means.csv` is retained for auditability.

---

## 23. Reproducibility and provenance

The repository distinguishes between:

### Directly preserved experimental configuration

These are supported by available source implementations and/or documented experimental settings, including:

- seed set;
- dataset/representation structure;
- model architectures;
- model hyperparameters;
- training procedure;
- preprocessing logic;
- metric definitions;
- representation construction details available from the source implementation.

### Reconstructed reproducibility components

Some repository modules have been organized from the documented manuscript methodology and available experimental information because the complete historical development environment is not being reconstructed.

These files are intended to provide a clean, executable representation of the documented experiment rather than to claim byte-for-byte identity with every historical notebook or development script.

### Not included

The repository does not include:

- obsolete development notebooks;
- temporary files;
- exploratory intermediate files;
- unrelated Colab artifacts;
- original benchmark datasets;
- unverified historical result files.

---

## 24. Supplementary optimization experiments

Representation-specific hyperparameter optimization analyses are treated separately from the controlled primary fixed-configuration experiment.

The supplementary optimization experiments should not be used to imply that architecture–representation compatibility has been independently established as a general property.

The primary conclusions concerning representation-dependent learning behaviour are based on the controlled experimental configuration.

---

## 25. Calibration interpretation

The reported ECE and Brier Score values are calibration measurements under the controlled class-balanced evaluation distribution used in the study.

They should not be interpreted as direct estimates of deployment-time calibration under naturally occurring intrusion prevalence.

Operational calibration assessment would require representative traffic reflecting the target environment's class prevalence and, where appropriate, deployment-specific recalibration.

---

## 26. Scope of interpretation

The common evaluated attack space is:

```text
Benign
SYN Flood
UDP Flood
```

Consequently, the experiments do not by themselves establish broad IDS generalization to:

- low-volume attacks;
- application-layer attacks;
- reconnaissance;
- credential attacks;
- malware;
- lateral movement;
- mixed multi-stage intrusions;
- other attack families not represented in the controlled evaluation.

Additional datasets, attack categories, traffic distributions, and deployment conditions would be required to evaluate such broader generalization.

---

## 27. Data availability statement

The datasets used in the study are publicly available from their original sources, including CIC-DDoS2019, CICIoT2023, and CIC-IoT-DIAD2024.

No new raw benchmark datasets are created by this repository.

The repository is intended to provide the reproducibility materials associated with the study, including:

- representation definitions;
- feature mappings where available;
- partition assignments;
- random seeds;
- preprocessing procedures;
- model configurations;
- evaluation procedures;
- calibration analysis;
- statistical analysis;
- sensitivity analysis;
- table-generation scripts.

The original benchmark datasets are not redistributed.

A final public repository URL should be added here only after the GitHub repository has actually been created and verified.

---

## 28. Recommended execution order

For a clean reproduction:

```text
1. Obtain the original benchmark datasets
                    ↓
2. Prepare the documented derived representations
                    ↓
3. Verify train/validation/test partitions
                    ↓
4. Run primary classical experiments
                    ↓
5. Run primary deep-learning experiments
                    ↓
6. Save run-level predictions/probabilities
                    ↓
7. Run evaluation and calibration analysis
                    ↓
8. Run statistical analysis
                    ↓
9. Run RSI analysis
                    ↓
10. Run sensitivity/audit analyses
                    ↓
11. Generate manuscript tables
                    ↓
12. Generate Figure 8 reliability diagrams
```

The run-level result files should be retained as the master experimental source for downstream analyses.

---

## 29. Citation

If this repository is used or adapted, please cite the associated manuscript:

> **Representation-Dependent Learning Behaviour in Machine Learning-Based Intrusion Detection: Experimental Evidence and the Representation-Dependent Discriminative Structure Principle**

The final citation details should be updated after publication.

---

## 30. Reproducibility principle

The repository follows one central rule:

> **Every reported aggregate should be traceable to condition-level experimental observations and, where applicable, individual run-level results.**

This means that:

```text
raw/derived data
      ↓
condition-level results
      ↓
run-level master results
      ↓
statistical analysis
      ↓
tables / figures
```

is preferred over manually entering or editing manuscript-level numerical results.
