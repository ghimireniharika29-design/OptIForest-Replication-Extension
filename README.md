# OptIForest: Replication and Extension

## Project Overview

This project replicates and extends *OptIForest: Optimal Isolation Forest for Anomaly Detection* (IJCAI 2023) as part of COMP8240 Applications of Data Science at Macquarie University.

The project aims to:

- Replicate the original OptIForest experiment using the AD benchmark dataset.
- Evaluate OptIForest on an additional real-world dataset (CMC).
- Construct synthetic datasets with different anomaly difficulty levels to investigate model performance.

**Original implementation:** https://github.com/xiagll/OptIForest

The original detector and demonstration script were retained. Additional preprocessing scripts, synthetic dataset generators and experiments were developed for this project.

## Environment

Experiments were conducted using Python 3.10.21 in GitHub Codespaces.

Main dependencies:

- NumPy 1.26.4
- pandas 2.2.3
- SciPy 1.12.0
- scikit-learn 1.4.2

These are the versions used in the replication environment. Additional dependencies may be required by the original implementation.

## Datasets

All datasets are stored as headerless CSV files, with binary anomaly labels in the final column (0 = normal, 1 = anomaly).

| Dataset | Observations | Features | Anomalies |
|---|---:|---:|---:|
| AD (original benchmark) | 3,279 | 1,555 | 459 |
| CMC (real-world extension) | 1,473 | 25 | 29 |
| Synthetic Easy (+4) | 2,000 | 10 | 100 |
| Synthetic Moderate (+2) | 2,000 | 10 | 100 |
| Synthetic Harder (+1) | 2,000 | 10 | 100 |

The CMC dataset was obtained from the [ADBenchmarks dataset collection](https://github.com/mala-lab/ADBenchmarks-anomaly-detection-datasets) and processed using `new_data/preprocess_cmc.py`.

The synthetic datasets were generated using `new_data/generate_synthetic.py` and `new_data/generate_synthetic_difficulty.py`, with random seed 42. The difficulty levels correspond to shifts of +4, +2 and +1 applied to the first three features of anomalous observations.

## Running the Experiments

Run these commands from the repository root in a compatible Python environment.

### Prepare the CMC Dataset

The CMC dataset is sourced from the ADBenchmarks anomaly detection dataset repository. After obtaining the original `cmc-nominal.arff` file and placing it in the directory expected by the preprocessing script, run:

```bash
python new_data/preprocess_cmc.py
```

The script one-hot encodes the categorical predictors and saves the processed dataset as `new_data/cmc_processed.csv`. To run the original OptIForest demonstration script, copy this file to `data/cmc.csv`.

### Generate Synthetic Datasets

```bash
python new_data/generate_synthetic.py
python new_data/generate_synthetic_difficulty.py
```

### Original AD Replication

```bash
python demo.py --dataset=ad --threshold=403 --branch=0
```

### CMC Dataset

```bash
python demo.py --dataset=cmc --threshold=55 --branch=0
```

### Synthetic Datasets

```bash
python demo.py --dataset=synthetic --threshold=55 --branch=0
python demo.py --dataset=synthetic_moderate --threshold=55 --branch=0
python demo.py --dataset=synthetic_hard --threshold=55 --branch=0
```

Each experiment uses 100 ensemble members and 15 repetitions, following the original `demo.py` procedure.

The script prints ROC AUC, Average Precision (reported as PR score), standard deviations, and average training and scoring times.

## Results

Saved outputs from the completed experiments are available in:

`experiments/results/`

The original `demo.py` prints results to the terminal. To save a new run, redirect its output using `tee`, for example:

```bash
python demo.py --dataset=ad --threshold=403 --branch=0 2>&1 | tee experiments/results/ad_t403_b0.log
```

The original evaluation fits and scores the same observations rather than using a separate held-out test set. Detector randomness is not explicitly seeded, so repeated runs may produce slightly different results.

Detailed experimental comparisons, interpretations, limitations and ethical considerations are presented in the accompanying final report.

## Attribution

This repository builds upon the original [OptIForest implementation](https://github.com/xiagll/OptIForest). The additional dataset preparation, synthetic data generation and extension experiments were undertaken for the COMP8240 project.

**Macquarie University — COMP8240 Applications of Data Science**