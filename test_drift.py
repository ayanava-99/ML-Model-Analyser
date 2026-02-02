import pandas as pd
import numpy as np
from drift_logic import calculate_drift_stats

# Create dummy data
# Reference: Normal dist mean 0
ref_data = pd.DataFrame({
    "numeric_col": np.random.normal(0, 1, 100),
    "categorical_col": ["A"]*80 + ["B"]*20
})

# Current: Normal dist mean 2 (Drift!)
curr_data = pd.DataFrame({
    "numeric_col": np.random.normal(2, 1, 100),
    "categorical_col": ["A"]*50 + ["B"]*40 + ["C"]*10 # Drift in props + New category
})

print("Running Drift Calculation...")
report = calculate_drift_stats(ref_data, curr_data)

print("\n--- Numerical Drift ---")
for res in report["numerical_drift"]:
    print(f"Feature: {res['feature']}, Mean Diff: {res['mean_diff']:.2f}, Drift Detected: {res['drift_detected']}")

print("\n--- Categorical Drift ---")
for res in report["categorical_drift"]:
    print(f"Feature: {res['feature']}, Max Prop Diff: {res['max_prop_diff']:.2f}, New Cats: {res['new_categories']}, Drift Detected: {res['drift_detected']}")

assert report["numerical_drift"][0]["drift_detected"] == True
assert report["categorical_drift"][0]["drift_detected"] == True
print("\nVerfication PASSED!")
