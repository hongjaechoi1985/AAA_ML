import os
import numpy as np
import pandas as pd
import joblib
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, brier_score
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from lifelines import WeibullAFTFitter, LogNormalAFTFitter
from sksurv.ensemble import ExtraSurvivalTrees
# from pycox.models import CoxBoost
# from xgbse.contrib import XGBSEDebiased
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

from functools import partial
from sklearn.inspection import permutation_importance
from sklearn.utils import shuffle

#=============================================
#1. Lata load
#=============================================
today = 20250219


file_path = 'data/data_out/data_store/Final_cohort/result_cohort/All_15065_result1.csv'
data = pd.read_csv(file_path)
data = data[data['OP_type'] == 4.0].reset_index(drop=True)

def is_malignant(code):
    code = code.strip()
    if not code.startswith('C'):
        return False
    
    num_part = code[1:]
    try:
        num_val = int(num_part)
        return 0 <= num_val <= 97
    except ValueError:
        return False
    
copd_codes = ['J44']
cvd_codess = ['I60', 'I61', 'I62', 'I63', 'I64']
mi_codes = ['I21', 'I22']

def parse_codes(code_str):
    return [code.strip() for code in code_str.split(',')]

data['COPD'] = data['MCEX_SICK_SYM'].apply(lambda x: 1 if any(code in copd_codes for code in parse_codes(x)) else 0)
data['CVD'] =  data['MCEX_SICK_SYM'].apply(lambda x: 1 if any(code in cvd_codess for code in parse_codes(x)) else 0)
data['MI'] = data['MCEX_SICK_SYM'].apply(lambda x: 1 if any(code in mi_codes for code in parse_codes(x)) else 0)
data['Cancer'] = data['MCEX_SICK_SYM'].apply(lambda x: 1 if any(is_malignant(code) for code in parse_codes(x)) else 0)

##Machine learning

feat_list = ['Age','G1E_BMI', 'G1E_BP_SYS',
            'G1E_BP_DIA', 'G1E_HGB','G1E_FBS', 'G1E_TOT_CHOL', 'G1E_SGOT', 'G1E_SGPT', 'G1E_GGT','G1E_WSTC', 
            'G1E_TG', 'G1E_HDL', 'G1E_LDL', 'G1E_CRTN', 'G1E_GFR','D_O_duration',
             'Q_SMK_YN',  
            'SEX_TYPE', 'Insurance', 'new_area', 'Reg_ex', 
             'Q_FHX_HTN', 'Q_FHX_DM', 'CAD', 'DM', 'Dyslipidemia', 'HTN', 'CKD', 'CVD','Cancer',
             #'OP_type','new_admit','Alcohol','G1E_HGHT', 'G1E_WGHT'
            'Death_AA','duration']

num_list =['Age', 'G1E_WGHT','G1E_HGHT','G1E_BMI','G1E_WSTC','D_O_duration', 'G1E_BP_SYS',  'G1E_BP_DIA', 
           'G1E_HGB','G1E_FBS', 'G1E_TOT_CHOL', 'G1E_SGOT', 'G1E_SGPT','G1E_TG', 'G1E_GGT', 
           'G1E_HDL', 'G1E_LDL', 'G1E_CRTN', 'G1E_GFR',] #

category_list =['Q_SMK_YN','new_SES','mapped_living_area'] #'new_admit','Alcohol',

bin_list =[ 'Q_FHX_HTN','Q_FHX_DM',
           'HTN', 'DM', 'Dyslipidemia', 'CAD', 'CKD', 'CVD','Cancer',
           'SEX_TYPE', 'Insurance', 'Reg_ex',  ]#'OP_type''new_area',

label_list =['Death_AA','duration']

# (1) Numeric
MM_scaler = MinMaxScaler()
numeric_variable_scale = MM_scaler.fit_transform(data[num_list])
numeric_variable_scale_df = pd.DataFrame(numeric_variable_scale, columns=num_list)
numeric_variable_scale_df = numeric_variable_scale_df.fillna(numeric_variable_scale_df.mean())

# (2) Category
for column in category_list:
    data[column] = data[column].astype('object')
for column in bin_list:
    data[column] = data[column].astype('object')
    
imputer_cat = SimpleImputer(strategy='most_frequent')
category_variable_df = pd.DataFrame(imputer_cat.fit_transform(data[category_list]), columns=category_list)

#Ont-hot Encoding
category_encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
category_encoded = category_encoder.fit_transform(category_variable_df)
category_encoded_df = pd.DataFrame(category_encoded, columns=category_encoder.get_feature_names_out(category_list))

# (3) Binary
imputer_bin = SimpleImputer(strategy='most_frequent')
bin_variable_df = pd.DataFrame(imputer_bin.fit_transform(data[bin_list]), columns=bin_list)

# (4) Label
label_variable_df = data[label_list].copy()
label_variable_df['Death_AA'] = label_variable_df['Death_AA'].apply(lambda x: 1 if x ==1 else 0) 

# (5) Data concat
data_cleaned = pd.concat([numeric_variable_scale_df,
                             bin_variable_df,
                             category_variable_df,
                             label_variable_df],
                             axis=1)

# (6) Preprocess, reproductive
feature_names = data_cleaned.drop(columns=label_list).columns.tolist()
joblib.dump(feature_names, f'feature_names_{today}.joblib') ##추후 Permutation Importance에서 사용
joblib.dump(MM_scaler, f'minmax_scaler_{today}.joblib')
joblib.dump(imputer_cat, f'category_imputer_{today}.joblib')
joblib.dump(imputer_bin, f'binary_imputer_{today}.joblib')
joblib.dump(category_encoder, f'onehot_encoder_{today}.joblib')

# (7) save
data_cleaned.to_csv(f'data_cleaned_{today}.csv', index=False)



#==================================================== IPCW
#2. feature, label modified
#====================================================

X = data_cleaned.drop(columns=['Death_AA', 'duration'])
y = data_cleaned[['Death_AA', 'duration']].rename(columns={'Death_AA': 'event'})
y['event'] = y['event'].astype(bool)
y['duration'] = y['duration'].apply(lambda x: x if x > 0 else 1e-6)

y_surv_full = Surv.from_dataframe('event', 'duration', y)

#Data split(hold-out)
X_train, X_test, y_train_df, y_test_df = train_test_split(X, y, test_size=0.2, stratify=y['event'], random_state=42)
y_train = Surv.from_dataframe('event', 'duration', y_train_df)
y_test = Surv.from_dataframe('event', 'duration', y_test_df)

y_train_structured = np.array(list(zip(y_train_df['event'], y_train_df['duration'])),
                             dtype=[('event', '?'), ('duration', '<f8')])
y_test_structured = np.array(list(zip(y_test_df['event'], y_test_df['duration'])),
                             dtype=[('event', '?'), ('duration', '<f8')])

#=====================================================
#3. Model and evaluation
#=====================================================
model_templates = {
    'RSF': lambda: RandomSurvivalForest(
        n_estimators=200, min_samples_split=10, min_samples_leaf=7, 
        max_features='sqrt', n_jobs=-1, random_state=42),
    'GBSA': lambda: GradientBoostingSurvivalAnalysis(random_state=42),
    'CoxPH': lambda: CoxPHSurvivalAnalysis(alpha=0.01),
    'EXT' : lambda: ExtraSurvivalTrees(n_estimators=200, min_samples_split=10, min_samples_leaf=7, 
        max_features='sqrt', n_jobs=-1, random_state=42)
}

eval_times = np.array([30, 90, 180, 270, 365, 730, 1095])



#===========================================================
#4. Stratified 5-fold CV on Train, c-index
#===========================================================

skf_train = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_c_index = {model_name: {tau: [] for tau in eval_times} for model_name in model_templates.keys()}
brier_score_results = {model_name: {tau: [] for tau in eval_times} for model_name in model_templates.keys()}

for fold, (train_idx, valid_idx) in enumerate(skf_train.split(X_train, y_train_df['event'])):
    print(f'\n[CV Fold {fold + 1}]')
    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[valid_idx]
    
    y_tr_df = y_train_df.iloc[train_idx]
    y_val_df = y_train_df.iloc[valid_idx]
    y_tr = Surv.from_dataframe('event', 'duration', y_tr_df)
    y_val = Surv.from_dataframe('event', 'duration', y_val_df)
    
    y_val_structured = np.array(list(zip(y_val_df['event'], y_val_df['duration'])),
                                dtype=[('event', '?'), ('duration', '<f8')])
    
    for model_name, model_factory in model_templates.items():
        model = model_factory()
        model.fit(X_tr, y_tr)
        
        surv_funcs = model.predict_survival_function(X_val)
        pred_surv_matrix = np.array([[fn(t) for t in eval_times] for fn in surv_funcs])
        
        for t_idx, tau in enumerate(eval_times):
            pred_risk = 1.0 - pred_surv_matrix[:, t_idx]
            c_index_val = concordance_index_ipcw(y_tr, y_val_structured, pred_risk, tau=tau)[0]
            cv_c_index[model_name][tau].append(c_index_val)
            print(f"{model_name} - time {tau}: c-index = {c_index_val:.4f}")
        
        # Brier score 계산
        # brier_score 함수는 (times, scores) 반환 (scores: 각 eval_time에 대한 brier score)
        times_cv, brier_scores = brier_score(y_tr, y_val, pred_surv_matrix, eval_times)
        for t_idx, tau in enumerate(eval_times):
            brier_score_results[model_name][tau].append(brier_scores[t_idx])

            
            
#=============================================================
#5. Mean CV c-index
#=============================================================
print("\n=== Mean C-index 및 95% CI ===")
averaged_c_index = {model_name: {} for model_name in model_templates.keys()}
cindex_summary = []  

for model_name, tau_dict in cv_c_index.items():
    for tau in eval_times:
        values = np.array(tau_dict[tau])
        if len(values) == 0:
            continue
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        n = len(values)
        t_val = stats.t.ppf(1 - 0.025, df=n - 1)
        ci_lower = mean_val - t_val * std_val / np.sqrt(n)
        ci_upper = mean_val + t_val * std_val / np.sqrt(n)
        averaged_c_index[model_name][tau] = mean_val
        cindex_summary.append({
            'Model': model_name,
            'Eval_time': tau,
            'Mean_C_index': mean_val,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper
        })
        print(f" evaluation_time {tau}: Mean C-index = {mean_val:.4f}, 95% CI = ({ci_lower:.4f}, {ci_upper:.4f})")

# Mean Brier Score
print("\n=== Mean Brier Score ===")
averaged_brier = {model_name: {} for model_name in model_templates.keys()}
brier_summary = [] 

for model_name, tau_dict in brier_score_results.items():
    for tau in eval_times:
        values = np.array(tau_dict[tau])
        if len(values) == 0:
            continue
        mean_brier = np.mean(values)
        averaged_brier[model_name][tau] = mean_brier
        brier_summary.append({
            'Model': model_name,
            'Eval_time': tau,
            'Mean_Brier_Score': mean_brier
        })
        print(f" evaluation_time {tau}: Mean Brier Score = {mean_brier:.4f}")

model_list = list(model_templates.keys())
weights_by_time = {}
weights_summary = []  

for tau in eval_times:
    c_vals = [averaged_c_index[m].get(tau, 0) for m in model_list]
    sum_c = sum(c_vals)
    if sum_c == 0:
        weights = [1.0 / len(model_list)] * len(model_list)
    else:
        weights = [cv / sum_c for cv in c_vals]
    weights_by_time[tau] = weights
   
    for m_idx, model in enumerate(model_list):
        weights_summary.append({
            'Eval_time': tau,
            'Model': model,
            'Weight': weights[m_idx]
        })
    print(f"\n[Time {tau}] Model Weights: {dict(zip(model_list, weights))}")

# ----- CSV  -----
results_dir = 'AAA_ML_results'
os.makedirs(results_dir, exist_ok=True)

# 1. Mean C-index, CI
df_cindex_summary = pd.DataFrame(cindex_summary)
file_cindex = f"{results_dir}/averaged_c_index_summary.csv"
df_cindex_summary.to_csv(file_cindex, index=False)
print(f"\n mean C-index 및 95% CI: {file_cindex}")

# 2. Mean Brier Score 
df_brier_summary = pd.DataFrame(brier_summary)
file_brier = f"{results_dir}/averaged_brier_score_summary.csv"
df_brier_summary.to_csv(file_brier, index=False)
print(f"mean Brier Score: {file_brier}")

# 3. Time-varying weights
df_weights = pd.DataFrame(weights_summary)
file_weights = f"{results_dir}/dynamic_weights_by_time.csv"
df_weights.to_csv(file_weights, index=False)
print(f"Time-varying weights: {file_weights}")
    
#================================================================
#6. Full Train 
#================================================================
pred_surv_test = {}

for model_name, model_factory in model_templates.items():
    model = model_factory()
    model.fit(X_train, y_train)
    
    surv_funcs_test = model.predict_survival_function(X_test)
    pred_surv_matrix_test = np.array([[fn(t) for t in eval_times] for fn in surv_funcs_test])
    pred_surv_test[model_name] = pred_surv_matrix_test
    
pred_surv_ensemble = np.zeros((len(X_test), len(eval_times)))
for t_idx, tau in enumerate(eval_times):
    weights = weights_by_time[tau]
    
    for i in range(len(X_test)):
        surv_ens = 0.0
        for m_idx, m in enumerate(model_list):
            surv_ens += weights[m_idx] * pred_surv_test[m][i, t_idx]
        pred_surv_ensemble[i, t_idx] = surv_ens
        
#==================================================================
#7. Evaluation
#==================================================================
#Brier socre
times, ensemble_brier_scores = brier_score(y_train, y_test, pred_surv_ensemble, eval_times)

ensemble_c_index = {}
for t_idx, tau in enumerate(eval_times):
    risk_ensemble = 1.0 - pred_surv_ensemble[:, t_idx]
    c_index_val = concordance_index_ipcw(y_train, y_test_structured, risk_ensemble, tau=tau)[0]
    ensemble_c_index[tau] = c_index_val
    
print("Brier score:")
for t, bs in zip(times, ensemble_brier_scores):
    print(f" Time{t}: {bs:.4f}")
    
print("C-index:")
for t, c_val in ensemble_c_index.items():
    print(f" Time{t}: {c_val:.4f}")


#===========================================================
#8. Dynamic Ensemble
#===========================================================

class DynamicEnsembleSurvival:
    def __init__(self, base_models, weights_by_time, eval_times):
        self.base_models = base_models
        self.weights_by_time = weights_by_time
        self.eval_times = eval_times
        self.model_list = list(base_models.keys())
        
    def predict_survival_function(self, X):
        n_samples = X.shape[0]
        pred_surv_ensemble = np.zeros((n_samples, len(self.eval_times)))
        for t_idx, tau in enumerate(self.eval_times):
            weights = self.weights_by_time[tau]
            ensemble_preds = np.zeros(n_samples)
            for m_idx, m in enumerate(self.model_list):
                surv_funcs = self.base_models[m].predict_survival_function(X)
                preds = np.array([fn(tau) for fn in surv_funcs])
                ensemble_preds += weights[m_idx] * preds
            pred_surv_ensemble[:, t_idx] = ensemble_preds
        return pred_surv_ensemble
    
    def predict(self, X, time_point):
        surv_matrix = self.predict_survival_function(X)
        idx = np.argmin(np.abs(self.eval_times - time_point))
        return 1.0 - surv_matrix[:, idx]
    
    
    
    
#=================================================================
#9. Full train 
#=================================================================

base_models_full = {}
for model_name, model_factory in model_templates.items():
    model = model_factory()
    model.fit(X_train, y_train)
    base_models_full[model_name] = model
    
# Dynamic Enssemble 
ensemble_model = DynamicEnsembleSurvival(base_models_full, weights_by_time, eval_times)


#Model save
joblib.dump(ensemble_model, 'dynamic_ensemble_model.joblib')
print("Dynamic Ensemble model save done")



#=============================================================

###  Permutation
best_model = joblib.load('dynamic_ensemble_model.joblib')
print("Dynamic Ensemble ")
best_model.fit(X_train, y_train)



if best_model is not None:
    #Permutation
    print('\nPermutation Importance')
    
    feature_groups = {}
    for feature in num_list:
        feature_groups[feature] = [feature]
    for feature in bin_list:
        feature_groups[feature] = [feature]
    for feature in category_list:
        one_hot_cols = [col for col in data_cleaned.columns if col.startswith(feature + '_')]
        if not one_hot_cols:
            feature_groups[feature] = [feature]
        else:
            feature_groups[feature] = one_hot_cols
    
    # Permutation importance
    
    # Evaluation time
    eval_times = np.array([30, 90, 180, 270, 365, 730, 1095])
    
    #C-index
    def compute_c_index(y_train_structured, model, X_subset, y_subset, tau):
        try:
            if isinstance(model, (WeibullAFTFitter, LogNormalAFTFitter)):
                if isinstance(model, WeibullAFTFitter):
                    if hasattr(model, 'predict_log_partial_harzard'):
                        pred_log_partial_hazard = model.predict_log_partial_hazard(X_subset).values.ravel()
                        pred_risk = np.exp(pred_log_partial_hazard)
                    else:   
                        #predict_median사용
                        pred_median = model.predict_median(X_subset)
                        pred_risk = 1 / np.clip(pred_median.values, a_min=1e-6, a_max=None)
                elif isinstance(model, LogNormalAFTFitter):
                    pred_median = model.predict_median(X_subset)
                    pred_risk = 1 / np.clip(pred_median.values, a_min=1e-6, a_max=None)
            else:
                pred_risk = model.predict(X_subset, time_point=tau)
        
            
            y_subset_structured = np.array(list(zip(y_subset['event'], y_subset['duration'])),
                                           dtype=[('event', '?'), ('duration', '<f8')])
            
            #C-index
            c_index_result = concordance_index_ipcw(
                y_train_structured, y_subset_structured, pred_risk, tau=tau)
            c_index = c_index_result[0]
            return c_index
        except Exception as e:
            print(f'C-index at tau={tau}: {e}')
            return np.nan
    
    # Surv 
    y_surv_full = Surv.from_dataframe('event', 'duration', y)
    y_structured_full = np.array(list(zip(y['event'], y['duration'])),
                                 dtype=[('event', '?'), ('duration', '<f8')])
    
    
    original_c_indices = []
    for tau in eval_times:
        c_index = compute_c_index(y_structured_full, best_model, X, y, tau)
        original_c_indices.append(c_index)
          
     
    permutation_importance_list = []
    
    n_repeats = 5
    
    for feature, cols in feature_groups.items():
        feature_importance = {'Feature': feature}
        for eval_idx, tau in enumerate(eval_times):
            imp_scores = []
            for repeat in range(n_repeats):
                X_permuted = X.copy()
                if len(cols) == 1:
                    X_permuted[cols[0]] = shuffle(X_permuted[cols[0]].values, random_state=np.random.randint(0, 10000))
                else:
                    shuffled_indices = np.random.permutation(X_permuted.index)
                    X_permuted.loc[:, cols] =  X_permuted.loc[shuffled_indices, cols].values
                permuted_c_indices = compute_c_index(y_structured_full, best_model, X_permuted, y, tau)
                imp = original_c_indices[eval_idx] - permuted_c_indices
                imp_scores.append(imp)
                
            
            # Mean, SD
            avg_imp = np.mean(imp_scores)
            std_imp = np.std(imp_scores, ddof=1)
            n = len(imp_scores)
            t_val = stats.t.ppf(1 - 0.025, df=n - 1)
            ci_lower = avg_imp - t_val * std_imp / np.sqrt(n)
            ci_upper = avg_imp + t_val * std_imp / np.sqrt(n)
        
            # Mean, 95% CI
            feature_importance[f'Importance_{tau}'] = avg_imp
            feature_importance[f'CI_lower_{tau}'] = ci_lower
            feature_importance[f'CI_upper_{tau}'] = ci_upper            
            print(f'Feature {feature}, eval_time: {tau}, Importance: {avg_imp:.4f}, 95% CI: ({ci_lower:.4f}, {ci_upper:.4f})')
        print(f'Complete : {feature}\n')
        permutation_importance_list.append(feature_importance)
        
        
    importance_df = pd.DataFrame(permutation_importance_list)
    importance_df.to_csv('permutation_importance_by_eval_time.csv', index=False)
    
    print('Permutation Importance .')
else:
    print('can not calculation')
