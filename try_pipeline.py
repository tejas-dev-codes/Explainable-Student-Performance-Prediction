import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("=== Trying full pipeline from artefacts.pkl ===")
with open('results/artefacts.pkl', 'rb') as f:
    art = pickle.load(f)

preprocessor = art['preprocessor']
eng_scaler = art['eng_scaler']
rf_model = art['models']['Random Forest']

X_test_raw = art['X_test_raw']
y_test = art['y_test']

# Transform with preprocessor
print("Transforming with ColumnTransformer...")
X_base = preprocessor.transform(X_test_raw)
print(f"X_base shape: {X_base.shape}")
print(f"X_base dtype: {X_base.dtype}")

# Add engineered features
eng_features_names = art['eng_features']
print(f"\nEngineered features: {eng_features_names}")

# Compute engineered features from raw data
X_test_engineered = pd.DataFrame({
    'study_efficiency': X_test_raw['studytime'] / (X_test_raw['failures'] + 1),
    'academic_risk_score': X_test_raw['failures'] * 2 + X_test_raw['absences'] / 10 + (5 - X_test_raw['studytime']),
    'studytime_x_failures': X_test_raw['studytime'] * X_test_raw['failures'],
    'social_engagement': X_test_raw['goout'] + X_test_raw['freetime'] - X_test_raw['Dalc'] - X_test_raw['Walc'],
    'parent_edu_avg': (X_test_raw['Medu'] + X_test_raw['Fedu']) / 2,
    'support_index': (X_test_raw['schoolsup'] == 'yes').astype(int) + (X_test_raw['famsup'] == 'yes').astype(int) + (X_test_raw['paid'] == 'yes').astype(int),
    'health_x_absences': X_test_raw['health'] * X_test_raw['absences']
})
print(f"Engineered features shape: {X_test_engineered.shape}")

# Scale engineered features
X_engineered_scaled = eng_scaler.transform(X_test_engineered)
print(f"Engineered scaled shape: {X_engineered_scaled.shape}")

# Combine
X_final = np.hstack([X_base, X_engineered_scaled])
print(f"Final X shape: {X_final.shape}")

# Predict
with np.errstate(all='ignore'):
    y_pred = rf_model.predict(X_final)
    y_proba = rf_model.predict_proba(X_final)[:, 1]
print(f"\nPredictions shape: {y_pred.shape}")
print(f"Accuracy: {(y_pred == y_test).mean():.4f}")
print(f"Sample predictions: {y_pred[:5]}")
print(f"Sample true: {y_test.values[:5]}")
print(f"Sample proba: {y_proba[:5]}")

# Try single row
single_idx = 0
X_single_base = preprocessor.transform(X_test_raw.iloc[[single_idx]])
X_single_eng = pd.DataFrame({
    'study_efficiency': X_test_raw.iloc[[single_idx]]['studytime'] / (X_test_raw.iloc[[single_idx]]['failures'] + 1),
    'academic_risk_score': X_test_raw.iloc[[single_idx]]['failures'] * 2 + X_test_raw.iloc[[single_idx]]['absences'] / 10 + (5 - X_test_raw.iloc[[single_idx]]['studytime']),
    'studytime_x_failures': X_test_raw.iloc[[single_idx]]['studytime'] * X_test_raw.iloc[[single_idx]]['failures'],
    'social_engagement': X_test_raw.iloc[[single_idx]]['goout'] + X_test_raw.iloc[[single_idx]]['freetime'] - X_test_raw.iloc[[single_idx]]['Dalc'] - X_test_raw.iloc[[single_idx]]['Walc'],
    'parent_edu_avg': (X_test_raw.iloc[[single_idx]]['Medu'] + X_test_raw.iloc[[single_idx]]['Fedu']) / 2,
    'support_index': ((X_test_raw.iloc[[single_idx]]['schoolsup'] == 'yes').astype(int) + 
                     (X_test_raw.iloc[[single_idx]]['famsup'] == 'yes').astype(int) + 
                     (X_test_raw.iloc[[single_idx]]['paid'] == 'yes').astype(int)),
    'health_x_absences': X_test_raw.iloc[[single_idx]]['health'] * X_test_raw.iloc[[single_idx]]['absences']
})
X_single_eng_scaled = eng_scaler.transform(X_single_eng)
X_single_final = np.hstack([X_single_base, X_single_eng_scaled])
single_pred = rf_model.predict(X_single_final)[0]
single_proba = rf_model.predict_proba(X_single_final)[0][1]
print(f"\nSingle row - Prediction: {single_pred}, Probability of PASS: {single_proba:.4f}")
print(f"Single row - True: {y_test.iloc[single_idx]}")

print("\n=== SUCCESS: Full pipeline works! ===")