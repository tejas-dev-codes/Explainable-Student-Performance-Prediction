import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== Trying to load RandomForest from artefacts.pkl ===")
with open('results/artefacts.pkl', 'rb') as f:
    art = pickle.load(f)

rf_model = art['models']['Random Forest']
print(f"RandomForest type: {type(rf_model).__name__}")
print(f"RandomForest n_estimators: {rf_model.n_estimators}")
print(f"RandomForest max_depth: {rf_model.max_depth}")

# Try predicting with X_test_raw
X_test_raw = art['X_test_raw']
print(f"\nX_test_raw shape: {X_test_raw.shape}")
print(f"X_test_raw columns: {list(X_test_raw.columns)}")

# Test prediction
with np.errstate(all='ignore'):
    y_pred = rf_model.predict(X_test_raw)
    y_proba = rf_model.predict_proba(X_test_raw)[:, 1]
print(f"Predictions shape: {y_pred.shape}")
print(f"Predictions sample: {y_pred[:5]}")
print(f"Proba sample: {y_proba[:5]}")

# Test with a single row
single_row = X_test_raw.iloc[[0]]
pred = rf_model.predict(single_row)
proba = rf_model.predict_proba(single_row)
print(f"\nSingle row prediction: {pred[0]}")
print(f"Single row proba: {proba[0]}")

print("\n=== SUCCESS: RandomForest from artefacts.pkl works! ===")