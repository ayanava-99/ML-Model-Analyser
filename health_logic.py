import pandas as pd
import numpy as np
import google.generativeai as genai
import os

def check_quality(df):
    """
    Run comprehensive quality checks on the dataframe.
    """
    results = {
        "missing": {},
        "cardinality": {},
        "imbalance": {},
        "outliers": {},
        "stability": {},
        "summary": {
            "rows": len(df),
            "cols": len(df.columns),
            "total_cells": df.size,
            "total_missing": df.isna().sum().sum()
        }
    }
    
    # Identify columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # 1. Missing Values
    missing_series = df.isna().mean()
    for col, pct in missing_series.items():
        if pct > 0:
            flag = pct > 0.10
            count = int(df[col].isna().sum())
            results["missing"][col] = {"pct": pct, "count": count, "flagged": flag}

    # 2. Cardinality (Categorical)
    for col in cat_cols:
        unique_count = df[col].nunique()
        flag = unique_count > (len(df) * 0.9) and len(df) > 20 # Almost unique labels
        results["cardinality"][col] = {"count": unique_count, "flagged": flag}
        
    # 3. Imbalance (Categorical)
    for col in cat_cols:
        if df[col].dropna().empty:
            continue
        top_pct = df[col].value_counts(normalize=True).iloc[0]
        flag = top_pct > 0.90
        results["imbalance"][col] = {"top_pct": top_pct, "flagged": flag}
        
    # 4. Outliers (Numerical - IQR) & Stability
    for col in num_cols:
        data = df[col].dropna()
        if data.empty:
            continue
            
        # Stability (Constant check)
        std_dev = data.std()
        is_constant = std_dev == 0
        results["stability"][col] = {
            "mean": data.mean(),
            "std": std_dev,
            "min": data.min(),
            "max": data.max(),
            "flagged": is_constant # Flag if constant
        }
        
        # Outliers
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_pct = len(outliers) / len(data) if len(data) > 0 else 0
        
        flag = outlier_pct > 0.05 # Flag if > 5% are outliers
        results["outliers"][col] = {"pct": outlier_pct, "count": len(outliers), "flagged": flag}
        
    return results

def calculate_health_score(results):
    """
    Compute a 0-100 health score based on flagged issues.
    """
    score = 100
    penalties = 0
    
    # Penalties
    # Missing: -5 per flagged column
    for info in results["missing"].values():
        if info["flagged"]: penalties += 5
        
    # Cardinality: -2 per flagged
    for info in results["cardinality"].values():
        if info["flagged"]: penalties += 2
        
    # Imbalance: -3 per flagged
    for info in results["imbalance"].values():
        if info["flagged"]: penalties += 3
        
    # Outliers: -3 per flagged
    for info in results["outliers"].values():
        if info["flagged"]: penalties += 3
        
    # Constant features: -10 (useless)
    for info in results["stability"].values():
        if info["flagged"]: penalties += 10
        
    final_score = max(0, score - penalties)
    
    # Determine Status
    if final_score >= 85:
        status = "🟢 Healthy"
    elif final_score >= 60:
        status = "🟡 Needs Attention"
    else:
        status = "🔴 High Risk"
        
    return final_score, status

def generate_health_summary(results, score, status, api_key_input=None, model_name="gemini-1.5-flash"):
    """
    LLM Summary of dataset health.
    """
    api_key = api_key_input or os.getenv("LLM_API_KEY")
    if not api_key:
        return "Error: No API Key provided."
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Collect top issues for the prompt
    issues = []
    
    # Add flagged items
    for col, info in results["missing"].items():
        if info["flagged"]: issues.append(f"Missing Values: {col} ({info['pct']:.1%})")
        
    for col, info in results["imbalance"].items():
        if info["flagged"]: issues.append(f"Class Imbalance: {col} (Top class {info['top_pct']:.1%})")
    
    for col, info in results["outliers"].items():
        if info["flagged"]: issues.append(f"Outliers: {col} ({info['pct']:.1%})")
        
    for col, info in results["stability"].items():
        if info["flagged"]: issues.append(f"Constant Feature: {col}")

    summary_stats = results['summary']
    
    prompt = f"""
sYou are a Senior Data Engineer. Analyze this Dataset Health Report.

CONTEXT:
---
Dataset: {summary_stats['rows']} rows, {summary_stats['cols']} cols.
Overall Missing Cells: {summary_stats['total_missing']}

Health Score: {score}/100 ({status})

Detected High-Risk Issues (Top flagged items):
{issues[:15]}
---

TASK:
1. Summarize the dataset's readiness for ML training.
2. Identify the biggest risks (e.g. leakage via IDs, bias via imbalance, noise via outliers).
3. Suggest concrete cleaning steps (imputation, dropping, resampling).
4. Mention any downstream model risks.

OUTPUT FORMAT (Markdown):
## 🏥 Readiness Summary
## ⚠️ Critical Risks
## 🧹 Recommended Cleaning
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"
