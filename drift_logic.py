import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import google.generativeai as genai
import os

def calculate_drift_stats(ref_df, curr_df):
    """
    Compute drift statistics between two dataframes.
    
    Args:
        ref_df (pd.DataFrame): Reference dataset.
        curr_df (pd.DataFrame): Current dataset.
        
    Returns:
        dict: comprehensive drift report
    """
    report = {
        "summary": {},
        "numerical_drift": [],
        "categorical_drift": []
    }
    
    # 1. Basic Summary
    report["summary"] = {
        "ref_shape": ref_df.shape,
        "curr_shape": curr_df.shape,
        "ref_missing": ref_df.isna().mean().mean(),
        "curr_missing": curr_df.isna().mean().mean()
    }

    # Identify columns
    num_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = ref_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Ensure columns exist in both (intersection)
    num_cols = [c for c in num_cols if c in curr_df.columns]
    cat_cols = [c for c in cat_cols if c in curr_df.columns]
    
    # 2. Numerical Drift
    for col in num_cols:
        ref_data = ref_df[col].dropna()
        curr_data = curr_df[col].dropna()
        
        if len(ref_data) == 0 or len(curr_data) == 0:
            continue
            
        mean_diff = curr_data.mean() - ref_data.mean()
        std_diff = curr_data.std() - ref_data.std()
        
        # KS Test
        # Null hypothesis: samples are from same distribution.
        # Low p-value (< 0.05) -> Reject Null -> Drift detected.
        try:
            ks_stat, p_value = ks_2samp(ref_data, curr_data)
        except Exception:
            ks_stat, p_value = 0.0, 1.0

        is_drift = p_value < 0.05
        
        report["numerical_drift"].append({
            "feature": col,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "p_value": p_value,
            "drift_detected": is_drift
        })
        
    # 3. Categorical Drift
    for col in cat_cols:
        ref_counts = ref_df[col].value_counts(normalize=True)
        curr_counts = curr_df[col].value_counts(normalize=True)
        
        # Check for new or missing categories
        ref_cats = set(ref_counts.index)
        curr_cats = set(curr_counts.index)
        
        new_cats = list(curr_cats - ref_cats)
        missing_cats = list(ref_cats - curr_cats)
        
        # Simple drift metric: Max absolute difference in proportions
        all_cats = ref_cats.union(curr_cats)
        max_diff = 0.0
        for cat in all_cats:
            p_ref = ref_counts.get(cat, 0)
            p_curr = curr_counts.get(cat, 0)
            diff = abs(p_curr - p_ref)
            if diff > max_diff:
                max_diff = diff
        
        # Threshold: if any category proportion changes by > 10%
        is_drift = max_diff > 0.10 or len(new_cats) > 0
        
        report["categorical_drift"].append({
            "feature": col,
            "max_prop_diff": max_diff,
            "new_categories": new_cats,
            "missing_categories": missing_cats,
            "drift_detected": is_drift
        })
        
    return report

def explain_drift(drift_report, api_key_input=None, model_name="gemini-1.5-flash"):
    """
    Generate an LLM explanation for the drift report.
    """
    if not drift_report["numerical_drift"] and not drift_report["categorical_drift"]:
        return "No significant drift detected to explain."

    api_key = api_key_input or os.getenv("LLM_API_KEY")
    if not api_key:
        return "Error: No API Key provided."
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Filter for only drifting features to keep prompt short
    drifting_num = [d for d in drift_report["numerical_drift"] if d["drift_detected"]]
    drifting_cat = [d for d in drift_report["categorical_drift"] if d["drift_detected"]]
    
    summary = drift_report["summary"]
    
    prompt = f"""
You are a Senior ML Engineer. Analyze this Dataset Drift Report.

CONTEXT:
---
Shapes: Reference {summary['ref_shape']} -> Current {summary['curr_shape']}
Missing Data Ratio: Reference {summary['ref_missing']:.4f} -> Current {summary['curr_missing']:.4f}

Significant Numerical Drift (KS Test p < 0.05):
{drifting_num[:10]} (Top 10 shown)

Significant Categorical Drift (Prop Diff > 0.10 or New Cats):
{drifting_cat[:10]} (Top 10 shown)
---

TASK:
1. Summarize the major changes.
2. Classify the drift type (Covariate Drift, Prior Probability Shift, or Data Quality Issue).
3. Hypothesize real-world causes based on common ML pitfalls.
4. Recommend if retraining is necessary.

OUTPUT FORMAT:
Provide the response in Markdown.
## 🚨 Drift Summary
## 🧐 Analysis of Changes
## 🛠️ Recommended Actions
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"
