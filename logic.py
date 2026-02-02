import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def calculate_rules(metrics, stats):
    """
    Compute rule-based signals for model diagnosis.
    
    Args:
        metrics (dict): Dictionary containing training/val accuracy and loss.
        stats (dict): Dictionary containing dataset statistics.
        
    Returns:
        list[str]: A list of detected signals/issues.
    """
    signals = []
    
    train_acc = metrics.get('train_acc', 0)
    val_acc = metrics.get('val_acc', 0)
    
    # Heuristic 1: Overfitting
    # If training accuracy is significantly higher than validation accuracy (e.g., > 10% gap)
    if train_acc - val_acc > 0.10:
        signals.append(f"Potential Overfitting: Training accuracy ({train_acc:.2f}) is significantly higher than validation accuracy ({val_acc:.2f}).")
        
    # Heuristic 2: Underfitting
    # If both accuracies are low (threshold depends on problem, but let's assume < 0.60 as a generic signal for now, 
    # though in reality this is problem dependent. We'll stick to the gap being small but perf being low).
    # A better generic check might be if train_acc is low.
    if train_acc < 0.60 and val_acc < 0.60:
        signals.append(f"Potential Underfitting: Both training ({train_acc:.2f}) and validation ({val_acc:.2f}) accuracies are low.")

    # Heuristic 3: Class Imbalance
    # stats['class_dist'] is expected to be a dict or string representation of class %
    try:
        class_dist = stats.get('class_dist', {})
        if isinstance(class_dist, dict):
            for cls, pct in class_dist.items():
                if pct < 10.0:
                    signals.append(f"Class Imbalance: Class '{cls}' constitutes only {pct}% of the data (less than 10%).")
                    break # Report once
    except Exception:
        pass # Handle gracefully if format is unexpected

    # Heuristic 4: Small Data Regime
    num_samples = stats.get('num_samples', 0)
    num_features = stats.get('num_features', 0)
    
    if num_samples < 5000 and num_features > 50:
        signals.append(f"Small Data Regime / High Dimensionality: Only {num_samples} samples with {num_features} features. Risk of overfitting is high.")

    return signals

def diagnose_failure(metrics, stats, rules, api_key_input=None):
    """
    Uses LLM to provide a detailed diagnosis.
    """
    
    # Priority: Function arg (user input) -> Env var
    api_key = api_key_input or os.getenv("LLM_API_KEY")
    
    if not api_key:
        return "Error: No API Key provided. Please check your environment variables or sidebar input."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')

    # Construct the prompt
    prompt = f"""
You are a Senior Machine Learning Engineer expert in diagnosing model failures.
Analyze the following model training run and dataset statistics to diagnose why the model is underperforming.

CONTEXT:
---
Metrics:
{metrics}

Dataset Statistics:
{stats}

Rule-based Signals Detected:
{rules}
---

TASK:
1. Identify the most likely failure modes.
2. Explain WHY they are happening based on the evidence.
3. Suggest 3-5 concrete, practical remediation steps (engineering/data focused).
4. Mention any specific risks or caveats.

OUTPUT FORMAT:
Provide the response in the following Markdown structure (do NOT output preambles):

## 🔍 Likely Failure Causes
(Bullet points)

## 📉 Evidence from Metrics
(Explain how the numbers support the diagnosis)

## 🛠 Recommended Fixes
(Numbered list of concrete steps)

## ⚠️ Things to Watch Out For
(Caveats or risks)

Tone: Professional, concise, technical. No generic fluff.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"
