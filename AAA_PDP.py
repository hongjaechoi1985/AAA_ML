import joblib
import numpy as np

# dynamic stacking load
dynamic_model = joblib.load('dynamic_ensemble_model.joblib')
print("Dynamic stacking model loaded successfully.")

#  survival function prediction
pred_surv_funcs = dynamic_model.predict_survival_function(X)
if pred_surv_funcs is not None:
    try:
        pred_surv_matrix = np.array([
            [fn(t) for t in eval_times] for fn in pred_surv_funcs
        ])
        print(f'shape: {pred_surv_matrix.shape}')
    except Exception as e:
        print(f'error: {e}')
        pred_surv_matrix = None
else:
    print("Fail")
    pred_surv_matrix = None


def plot_risk_specific_values(model, scaler, X, feature, tau, specific_values=None, is_category=False):
    if is_category:
        print(f"Category '{feature}' with unique values: {specific_values}")
        # Use predefined values for categorical variables (can be modified as needed)
        specific_values = [1.0, 2.0, 3.0, 9.0]
        
        risk_list = []
        for val in specific_values:
            X_permuted = X.copy()
            X_permuted[feature] = val
            try:
                pred_funcs = model.predict_survival_function(X_permuted)
                surv_probs = np.array([fn(tau) for fn in pred_funcs])
                avg_surv_prob = surv_probs.mean()
                avg_risk = 1 - avg_surv_prob
                risk_list.append(avg_risk)
            except Exception as e:
                print(f'error {e}')
                risk_list.append(np.nan)
                
        plt.figure(figsize=(10, 6))
        plt.bar([str(val) for val in specific_values], risk_list, color='skyblue', edgecolor='black')
        plt.xlabel(f'{feature} (Category values)')
        plt.ylabel('Predicted Risk')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

        risk_df = pd.DataFrame({
            feature: [str(val) for val in specific_values],
            'Predict_risk': risk_list
        })
        risk_df.to_csv(f'risk_{feature}_tau_{tau}_category.csv', index=False)
        print(f"PDP 'risk_{feature}_tau_{tau}_category.csv' done.")
        return risk_df

    else:
        
        feature_min = scaler.data_min_[list(X.columns).index(feature)]
        feature_max = scaler.data_max_[list(X.columns).index(feature)]
        if specific_values is None:
            specific_values = np.linspace(feature_min, feature_max, num=10)
        
        specific_values = np.array(specific_values)
        print(f"Original '{feature}': {specific_values}")
        feature_grid_scaled = (specific_values - feature_min) / (feature_max - feature_min)
        feature_grid_scaled = np.clip(feature_grid_scaled, 0, 1)

        risk_list = []
        grid_points = len(specific_values)  
        for i, (orig_val, scaled_val) in enumerate(zip(specific_values, feature_grid_scaled)):
            X_permuted = X.copy()
            X_permuted[feature] = scaled_val
            try:
                pred_funcs = model.predict_survival_function(X_permuted)
                surv_probs = np.array([fn(tau) for fn in pred_funcs])
                avg_surv_prob = surv_probs.mean()
                avg_risk = 1 - avg_surv_prob
                risk_list.append(avg_risk)
                print(f"Grid Point {i+1}/{grid_points}: {orig_val}, Scaled value: {scaled_val}, Survival Probability: {avg_surv_prob}")
            except Exception as e:
                print(f'오류 {e}')
                risk_list.append(np.nan)

        plt.figure(figsize=(10, 6))
        plt.plot(specific_values, risk_list, marker='o', linestyle='-')
        plt.xlabel(f'{feature} (Original values)')
        plt.ylabel('Predicted Risk')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        risk_df = pd.DataFrame({
            feature: specific_values,
            'Predict_risk': risk_list
        })
        risk_df.to_csv(f'risk_{feature}_tau_{tau}.csv', index=False)
        print(f"PDP결과가 'risk_{feature}_tau_{tau}.csv'에 저장되었습니다.")
        return risk_df

# Example: performing a PDP for a specific SBP value.
specific_SBP_values = [90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
age_risk_df = plot_risk_specific_values(
    model=dynamic_model,   
    scaler=MM_scaler,
    X=X,
    feature='G1E_BP_SYS',
    tau=365,
    specific_values=specific_SBP_values,
    is_category=False
)
