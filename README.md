# PathSnap – Distributed Map-Matching & Route Reconstruction
**PES University | Department of Computer Science & Engineering**<br>
**Course:** UE24CS352A Machine Learning | **Hackathon 2026**

---

## 1. Project Overview
PathSnap is an end-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy, discretely sampled GPS observations onto physical road networks.

GPS tracking systems in urban environments inherently suffer from multipath interference, building occlusions, GPS dropouts, and low sampling frequencies. A vehicle traveling on an expressway or main street can appear to jump across buildings or onto parallel alleys. PathSnap solves this problem by combining:
1. **Spatial Candidate Generation & Geodesic Feature Engineering:** Identifying plausible road segment candidates within spatial query radiuses and computing geometric features (perpendicular distance, heading divergence, instantaneous velocity, projection ratios).
2. **Supervised Candidate Ranking (5 Model Families):** Training and benchmarking 5 distinct machine learning algorithm families using group-aware cross-validation to score candidate road segment likelihoods.
3. **Bayesian Hyperparameter Optimisation:** Fine-tuning the leading model using Optuna with group-aware cross-validation on 80 training trips.
4. **Sequential Route Reconstruction (HMM & Log-Viterbi):** Enforcing physical road network topology and transitions to output a continuous, valid driving route.
5. **Two-Laptop Distributed Deployment:** Hosting the inference model & server-side evaluation pipeline on Laptop 1 (FastAPI) and the interactive evaluator dashboard on Laptop 2 (Streamlit).

---

## 2. Dataset Partitions & Leakage Prevention
The system strictly enforces zero data leakage across partitions:
* **Training (80 trips):** Used exclusively for exploratory analysis, candidate extraction, and 5-fold `GroupKFold` cross-validation grouped strictly by `trip_id`.
* **Student Testing (10 trips):** Used exclusively for development-stage model evaluation, student Top-1 Accuracy benchmarking, and HMM parameter tuning.
* **Evaluator Testing (10 trips):** Kept strictly server-side on Laptop 1. Ground truth labels are never exposed to feature extraction, training pipelines, or client UI.

---

## 3. The 5 Benchmark Machine Learning Models
To meet the academic requirements of exploring multiple distinct algorithm families, PathSnap implements:

| Model | Algorithm Family | Configuration Highlights | Role in Pipeline |
|---|---|---|---|
| **Logistic Regression** | Linear & Probabilistic | Balanced class weights, L2 regularization, Standard Scaler | Calibrated probabilistic baseline |
| **Distance-Weighted KNN** | Instance-based | $k=15$, distance-weighted voting | Non-parametric spatial similarity |
| **Random Forest** | Ensemble (Bagging) | 150 de-correlated trees, balanced subsample | Robust non-linear interaction modeling |
| **HistGradientBoosting** | Gradient Boosting | Histogram binning, learning rate 0.08 | High-efficiency sequential error correction |
| **RBF SVM** | Kernel-based | Radial Basis Function kernel ($C=2.0$), Platt probability scaling | Maximum-margin decision boundary |

---

## 4. Repository Structure
```text
PathSnap/
├── README.md                          # Comprehensive setup & execution guide
├── requirements.txt                   # Python package dependencies
├── run_pipeline.py                    # Master execution CLI
│
├── config/
│   └── config.yaml                    # Global pipeline configurations
│
├── data/
│   ├── raw/                           # Raw partition directories (train, test)
│   └── processed/                     # Extracted tabular features (train, test)
│
├── src/
│   ├── data/
│   │   ├── loader.py                  # Robust readers for .track, .nodes, .arcs, .route
│   │   └── geo_utils.py               # Haversine & geodesic distance computations
│   ├── eda/
│   │   └── eda.py                     # Dataset inventory and statistical profiling
│   ├── features/
│   │   ├── candidate_generator.py     # KDTree spatial candidate queries
│   │   └── feature_extractor.py       # Geometric feature formulation
│   ├── models/
│   │   ├── train.py                   # 5-model training harness with GroupKFold
│   │   ├── evaluate.py                # Metric evaluation (Top-1 Accuracy, Macro F1, Log Loss)
│   │   └── bayesian_opt.py            # Optuna Bayesian hyperparameter search
│   ├── routing/                       # HMM & Log-Viterbi route reconstruction
│   └── visualization/
│       └── folium_maps.py             # Multi-layer Folium map generation
│
├── notebooks/
│   └── 01_eda_and_visualizations.ipynb # Step-by-step EDA walkthrough
│
├── artifacts/                         # Saved model weights (.joblib) & CV results
│   ├── training_cv_results.csv
│   └── *.joblib
│
└── outputs/
    ├── eda/                           # Inventory tables & statistical plots
    ├── maps/                          # Interactive Folium HTML maps
    ├── model_comparison/              # 5-model comparison table & selection report
    └── optimization/                  # Bayesian tuning study & comparison
```

---

## 5. Quickstart & Reproducibility Guide

### 1. Installation
Clone this repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis (EDA)
```bash
python run_pipeline.py --stage eda --data-root "D:\ML-HACKTHON\Student folder"
```

### 3. Run Feature Extraction
```bash
python run_pipeline.py --stage features --data-root "D:\ML-HACKTHON\Student folder"
```

### 4. Train & Evaluate the 5 Models
```bash
python run_pipeline.py --stage train
```

### 5. Run Bayesian Hyperparameter Optimisation
```bash
python run_pipeline.py --stage optimize
```

---

## 6. Phase 1 Deliverables Summary
* **EDA Inventory & Summary:** `outputs/eda/dataset_inventory.csv` & `outputs/eda/eda_summary.md`
* **Geographical Visualizations:** `outputs/maps/00000001_eda_map.html` & `outputs/eda/plots/*.png`
* **5-Model Benchmark Results:** `artifacts/training_cv_results.csv` & `outputs/model_comparison/model_comparison.csv`
* **Model Selection Justification:** `outputs/model_comparison/model_selection_report.md`

## Two-laptop evaluator demo

Keep the `evaluator_test/` partition on the server laptop. After training and
Optuna optimization, set `PATHSNAP_DATA_ROOT` to the directory containing
`evaluator_test/`, then start the API from the repository root:

```powershell
$env:PATHSNAP_DATA_ROOT = "C:\path\to\data-root"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Laptop 2 can list trip IDs with `GET http://<server-LAN-IP>:8000/trips` and
request a prediction with `POST /predict` and JSON `{"trip_id":"00000000"}`.
The response includes aggregate accuracy and predicted route arc IDs; reference
labels stay on Laptop 1. See `backend/README.md` for setup details.
