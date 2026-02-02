import pandas as pd
import numpy as np
from health_logic import check_quality, calculate_health_score

# Create dummy data
df_health = pd.DataFrame({
    "clean_num": np.random.normal(0, 1, 100),
    "dirty_num": np.append(np.random.normal(0, 1, 94), [100, 200, 300, 400, 500, 600]), # 6 outliers / 100 total > 5%
    "clean_cat": ["A"]*50 + ["B"]*50,
    "dirty_cat": ["A"]*95 + ["B"]*5, # Imbalance
    "missing_col": [1]*90 + [None]*10 # 10% missing
})

print("Running Quality Check...")
report = check_quality(df_health)

print("\n--- Missing ---")
print(report["missing"])

print("\n--- Outliers ---")
print(report["outliers"])

print("\n--- Imbalance ---")
print(report["imbalance"])

score, status = calculate_health_score(report)
# Remove emoji for Windows console compatibility
status_clean = status.replace("🟢", "").replace("🟡", "").replace("🔴", "").strip()
print(f"\nFinal Score: {score}, Status: {status_clean}")

assert report["outliers"]["dirty_num"]["flagged"] == True
assert report["imbalance"]["dirty_cat"]["flagged"] == True
assert score < 100
print("\nVerification PASSED!")
