from logic import calculate_rules

# Test Case 1: Overfitting
metrics = {"train_acc": 0.99, "val_acc": 0.60}
stats = {"num_samples": 1000, "num_features": 10}
rules = calculate_rules(metrics, stats)
print("Test 1 (Overfitting):", "Pass" if any("Overfitting" in r for r in rules) else "Fail")

# Test Case 2: Underfitting
metrics = {"train_acc": 0.55, "val_acc": 0.50}
rules = calculate_rules(metrics, stats)
print("Test 2 (Underfitting):", "Pass" if any("Underfitting" in r for r in rules) else "Fail")

# Test Case 3: Small Data
metrics = {"train_acc": 0.8, "val_acc": 0.7}
stats = {"num_samples": 100, "num_features": 60}
rules = calculate_rules(metrics, stats)
print("Test 3 (Small Data):", "Pass" if any("Small Data" in r for r in rules) else "Fail")
