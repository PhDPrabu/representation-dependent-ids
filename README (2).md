# Data Directory

This directory contains the **derived representation-level data** required by
the reproducibility repository.

The original benchmark datasets are **not redistributed** here.

## 1. Benchmark datasets

The study uses:

- CIC-DDoS2019
- CICIoT2023
- CIC-IoT-DIAD2024

Each dataset is treated as a separate experimental factor.

The representation factor is:

- Flow
- Header
- Hybrid

Dataset identity and representation identity must not be conflated.

---

## 2. Expected directory structure

```text
data/
├── CIC-DDoS2019/
│   ├── Flow/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   ├── Header/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── Hybrid/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── CICIoT2023/
│   ├── Flow/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   ├── Header/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── Hybrid/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
└── CIC-IoT-DIAD2024/
    ├── Flow/
    │   ├── train.csv
    │   ├── validation.csv
    │   └── test.csv
    ├── Header/
    │   ├── train.csv
    │   ├── validation.csv
    │   └── test.csv
    └── Hybrid/
        ├── train.csv
        ├── validation.csv
        └── test.csv
```

The exact filename aliases supported by the experiment runners are documented
in the corresponding code modules.

---

## 3. Required labels

The controlled classification setting uses three target classes:

```text
Benign
SYN_Flood
UDP_Flood
```

The available source implementations use dataset-specific label-column names.
The preprocessing code therefore supports:

```text
_class
Label
label
```

The label column must not be included as an input feature.

---

## 4. Metadata columns

Where present, the following source/provenance metadata are not treated as
predictive input features:

```text
_source_file
_class
Label
label
```

The `_source_file` field can be retained for provenance and duplicate/source
audits but must not be supplied to the predictive model.

---

## 5. CICIoT2023 representation construction

The available representation-construction source documents the following
controlled construction for CICIoT2023:

- 56,863 samples per target class;
- classes:
  - BenignTraffic
  - SYN_Flood
  - UDP_Flood;
- chunk size: 50,000;
- sampling seed: 2026;
- train/validation/test ratio:
  - 70%
  - 10%
  - 20%;
- group-aware handling of identical feature vectors;
- cross-partition duplicate verification.

The balanced target size is:

```text
56,863 × 3 = 170,589 rows
```

when all three classes are present at the specified sample count.

Representation construction does not apply normalization or imputation.
Those data-dependent transformations are handled later by the experimental
preprocessing pipeline.

---

## 6. CICIoT2023 Flow representation

The documented Flow feature group contains:

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

---

## 7. CICIoT2023 Header representation

The documented Header feature group contains:

```text
Header_Length
Protocol Type
Time_To_Live
fin_flag_number
syn_flag_number
rst_flag_number
psh_flag_number
ack_flag_number
ece_flag_number
cwr_flag_number
ack_count
syn_count
fin_count
rst_count
HTTP
HTTPS
DNS
Telnet
SMTP
SSH
IRC
TCP
UDP
DHCP
ARP
ICMP
IGMP
IPv
LLC
```

---

## 8. CICIoT2023 Hybrid representation

Hybrid combines the documented Header and Flow feature groups.

Therefore, for the available CICIoT2023 construction:

```text
Header: 30 features
Flow:   10 features
Hybrid: 40 features
```

The repository should preserve the documented feature ordering when the
corresponding representation files are constructed.

---

## 9. Other datasets

The repository does **not** fabricate feature mappings for CIC-DDoS2019 or
CIC-IoT-DIAD2024.

Where a complete dataset-specific representation mapping is not supported by
the available construction source, the representation-builder module raises
an explicit `NotImplementedError` rather than silently generating an assumed
mapping.

This is intentional: reproducibility requires the released mapping to be
traceable to the documented experimental source.

---

## 10. Preprocessing

Data-dependent preprocessing is fitted on the training partition only.

The documented procedure is:

```text
Training partition
      ↓
numeric conversion
      ↓
infinite values → missing
      ↓
median imputation fitted on training data
      ↓
Min-Max scaling fitted on training data
      ↓
transformation applied unchanged to validation/test
```

The validation and test partitions must not be used to estimate imputation or
scaling parameters.

For classical models, these operations are implemented inside the model
pipeline.

For deep learning, the repository provides the explicit training-fitted
preprocessing path.

---

## 11. Partition integrity

The following should be checked before running the primary experiments:

1. Train, validation, and test partitions exist.
2. Labels are present.
3. The same feature schema is used within a representation.
4. The label is excluded from the feature matrix.
5. `_source_file` is excluded from model input.
6. No non-finite values remain after preprocessing.
7. Data-dependent preprocessing is fitted only on training data.
8. Any documented duplicate/source-file checks have been completed.

---

## 12. Do not redistribute original benchmark data

The repository should contain only the derived/released materials permitted
for redistribution.

Do not place the original benchmark download archives into Git.

If a dataset's redistribution terms do not permit committing derived files,
keep the data locally and use the repository code with the corresponding
local paths.

---

## 13. Recommended Git handling

Large CSV files should not be committed blindly.

Before publishing the repository:

- check repository size;
- check the licensing/redistribution terms of each derived dataset;
- use Git LFS or an appropriate external data repository where necessary;
- record file hashes for released derived data;
- retain the partition and feature mappings required to reproduce the
  experimental configuration.

---

## 14. Data provenance

Each released derived dataset should, where permitted, be accompanied by:

```text
dataset name
representation
partition
feature mapping
sampling configuration
random seed
source/provenance metadata policy
```

The original benchmark datasets remain external dependencies of the
reproduction workflow.

---

## 15. Reproducibility principle

The data layer should remain traceable:

```text
Original benchmark dataset
          ↓
documented sampling / harmonization
          ↓
Flow / Header / Hybrid representation
          ↓
train / validation / test partitions
          ↓
primary experiment runner
```

No manuscript result value should be manually inserted into a dataset file.
