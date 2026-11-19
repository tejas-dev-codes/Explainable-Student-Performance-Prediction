import pickle
import numpy as np

with open('results/artefacts.pkl', 'rb') as f:
    art = pickle.load(f)

print("=== ARTEFACTS.PKL KEYS ===")
for k in art.keys():
    print(f"  {k}: {type(art[k]).__name__}")

print("\n=== BEST MODEL ===")
print(f"  best_model_name: {art['best_model_name']}")

print("\n=== MODELS ===")
for name, model in art['models'].items():
    print(f"  {name}: {type(model).__name__}")

print("\n=== FEATURE NAMES (38 total) ===")
for i, name in enumerate(art['feature_names']):
    print(f"  {i:2d}: {name}")

print("\n=== NUMERIC FEATURES ===")
for f in art['numeric_features']:
    print(f"  {f}")

print("\n=== CATEGORICAL FEATURES ===")
for f in art['categorical_features']:
    print(f"  {f}")

print("\n=== ENGINEERED FEATURES ===")
for f in art['eng_features']:
    print(f"  {f}")

print("\n=== METRICS ===")
print(art['metrics_df'])

print("\n=== Y TEST ===")
print(f"  y_test: {art['y_test'].shape}, dtype={art['y_test'].dtype}")

print("\n=== Y PROBA ===")
print(f"  y_proba: {art['y_proba'].shape}, dtype={art['y_proba'].dtype}")

print("\n=== X TEST RAW ===")
print(f"  X_test_raw: {art['X_test_raw'].shape}")
print(f"  columns: {list(art['X_test_raw'].columns)}")

print("\n=== SHAP VALUES ===")
print(f"  shap_values: {art['shap_values'].shape}")

print("\n=== FAIRNESS SUMMARY ===")
print(art['fairness_summary'])