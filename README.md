# AAA_ML

## 🛠 Environment & Dependencies

This research was conducted using **Python 3.8.16**. The main libraries required to run the analysis are categorized below:

### 1. Data Processing
* `numpy`, `pandas`: Data manipulation
* `scikit-learn`: Preprocessing (Imputer, Encoder, Scaler) and Model Selection

### 2. Survival Analysis (Key Libraries)
* `scikit-survival` (sksurv): CoxPH, RandomSurvivalForest, GradientBoosting, ExtraSurvivalTrees
* `lifelines`: Weibull and LogNormal AFT models

### 3. Evaluation & Visualization
* `concordance_index_ipcw`, `brier_score`: Model performance metrics
* `matplotlib`: Plotting results
* `joblib`: Model saving/loading

---

##  How to Install

You can install the necessary dependencies using pip:

```bash
pip install numpy pandas scikit-learn scikit-survival lifelines matplotlib joblib
