# ML Model Failure Diagnosis Engine

A practical tool for Machine Learning Engineers to diagnose model underperformance using training metrics, dataset statistics, and rule-based heuristics augmented by LLM reasoning.

![Screenshot of the app](https://via.placeholder.com/800x400?text=ML+Model+Failure+Diagnosis+Engine)

## Features

- **Rule-based Diagnostics**: Instantly detects common issues like Overfitting, Underfitting, Class Imbalance, and Small Data Regimes.
- **LLM-Powered Reasoning**: Uses an LLM (e.g., GPT-4o-mini) to provide a deep-dive analysis, explaining *why* the failure is happening and suggesting concrete engineering fixes.
- **Interactive UI**: Built with Streamlit for a clean, responsive experience.
- **Privacy Focused**: Metrics and stats are sent to the LLM, not your actual dataset.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ML_model_analyser
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

2.  **Configuration:**
    - You can provide your OpenAI API Key directly in the sidebar.
    - Alternatively, set it in your environment (or a `.env` file):
        ```bash
        export LLM_API_KEY='sk-...'
        ```

3.  **Perform Diagnosis:**
    - Enter your model's Training and Validation Accuracy/Loss.
    - Input basic dataset statistics (Samples, Features, Class Balance).
    - Click **Run Diagnosis**.

## Structure

- `app.py`: Main Streamlit application entry point.
- `logic.py`: Core logic for heuristics and LLM interaction.
- `requirements.txt`: Python dependencies.

## Example Scenario

**Input:**
- Train Accuracy: 99%
- Validation Accuracy: 65%
- Dataset: 2000 samples, 200 features.

**Diagnosis:**
- **Signal**: Overfitting detected.
- **Analysis**: Model is memorizing the small dataset due to high dimensionality relative to sample count.
- **Fixes**: Apply L2 regularization, use Dropout, or reduce feature space (PCA/Feature Selection).

---
*Built for real-world ML workflows.*
