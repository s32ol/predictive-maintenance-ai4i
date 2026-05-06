This project builds a binary classifier for predictive maintenance on the AI4I 2020 dataset (10,000 rows of synthetic CNC milling telemetry, 28:1 class imbalance). Two physics-aligned features (temp_diff, power_proxy) are engineered from EDA-surfaced multicollinearities, and an XGBoost model is benchmarked against a logistic regression baseline using stratified 5-fold CV. Final test PR-AUC is **0.865** (ROC-AUC 0.972) at a cost-tuned threshold τ=0.31, catching 56 of 68 failures (82% recall) at a 1.7% false-alarm rate. Every modeling decision traces back to a documented EDA finding; the 10:1 cost ratio and hyperparameters are flagged as production-calibratable assumptions.

# Predictive Maintenance — AI4I 2020

Multi-fault classification on industrial CNC milling telemetry. Predicts machine failure from sensor readings while remaining honest about a 28:1 class imbalance.

## Status

**Phases 1–3 complete.** Phase 4 (production framing & write-up) in progress.

## Headline results (Phase 3)

| Metric | XGBoost | Logistic Regression |
|---|---|---|
| Test PR-AUC | **0.865** | 0.51 |
| Test ROC-AUC | 0.972 | — |
| Recall @ τ=0.31 | 82% (56 / 68) | — |
| False-alarm rate | 1.7% | — |

XGBoost outperforms the LR baseline by **~1.7×** on PR-AUC. See `notebooks/02_modeling.ipynb` for the full walkthrough, the DS Deep Dive on why XGBoost wins, and the Phase 3 closeout.


## Problem

Industrial equipment fails for many distinct reasons — overheating, overstrain, power excursions, tool wear, and noise. Each failure mode has a different sensor signature and a different operational cost. The task is to distinguish a healthy run from a failing one in time to act, while remaining honest about the imbalance: roughly 3–4% of runs end in failure. A model that predicts "no failure" for every row is 96% accurate and zero percent useful.

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) (Matzka, 2020). 10,000 observations, five sensor channels (air temperature, process temperature, rotational speed, torque, tool wear), categorical product type (L/M/H), and five labeled failure modes (TWF, HDF, PWF, OSF, RNF) that roll up into a binary `Machine failure` flag.

Download `ai4i2020.csv` from the Kaggle link above and place it at `data/ai4i2020.csv`. The dataset is not committed to this repo (see `.gitignore`).

## Phase 1 findings

- **Class imbalance**: ~28:1 negative-to-positive (3.4% failure rate). Drives metric and split strategy.
- **Multicollinearity**: process temp ↔ air temp (r ≈ 0.88), torque ↔ rpm (r ≈ −0.88). By design — reflects real physics.
- **Failure rate by type**: L 3.9% / M 2.8% / H 2.1%. Type carries signal.
- **Schema clean**: no nulls, no duplicate IDs.
- **Label consistency**: `Machine failure = OR(TWF, HDF, PWF, OSF, RNF)`. The five mode flags are *components of the label* — they must be dropped from the feature set to prevent leakage.

See `notebooks/01_eda.ipynb` for the full walkthrough and figures.

## Repo structure

```
predictive-maintenance-ai4i/
├── data/                       # gitignored — drop ai4i2020.csv here
├── notebooks/
│   ├── 01_eda.ipynb            # Phase 1 EDA (executed, renders on GitHub)
│   ├── 01_eda.py               # Same notebook in jupytext .py format
│   └── 02_modeling.ipynb          # Phases 2–3 modeling (LR baseline + XGBoost, executed)
├── reports/figures/            # Generated EDA figures
├── src/
│   ├── data.py                 # Schema-validating loader
│   └── generate_synthetic.py   # Dev fallback (spec-faithful AI4I generator)
├── requirements.txt
└── README.md
```

## Setup

```bash
# clone
git clone https://github.com/s32ol/predictive-maintenance-ai4i.git
cd predictive-maintenance-ai4i

# install
pip install -r requirements.txt

# get the data (manual: download from the Kaggle link in the Dataset section)
# place it at data/ai4i2020.csv

# run the notebooks
jupyter lab notebooks/01_eda.ipynb       # Phase 1 — EDA
jupyter lab notebooks/02_modeling.ipynb  # Phases 2–3 — modeling
```

## Roadmap

- **Phase 1** ✓ — EDA: distributions, correlations, failure-mode breakdown, leakage check.
- **Phase 2** ✓ — Feature engineering (`temp_diff`, `power_proxy`), Type encoding, stratified split.
- **Phase 3** ✓ — Logistic-regression baseline; XGBoost classifier with `scale_pos_weight`. Evaluation on PR-AUC, ROC-AUC, recall, confusion matrix; cost-tuned threshold τ=0.31.
- **Phase 4** — Production framing (drift detection, calibration, tiered output, cost-weighted thresholds), write-up, notes on extending to a real-world fleet.

## Stack

Python · pandas · scikit-learn · XGBoost · matplotlib · seaborn · Jupyter
