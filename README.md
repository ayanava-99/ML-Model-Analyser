# ML Model Analyser

A practical tool for Machine Learning Engineers to diagnose model underperformance using training metrics, dataset statistics, and rule-based heuristics augmented by LLM reasoning.

![Screenshot of the app](https://via.placeholder.com/800x400?text=ML+Model+Failure+Diagnosis+Engine)

## Features

- **Rule-based Diagnostics**: Instantly detects common issues like Overfitting, Underfitting, Class Imbalance, and Small Data Regimes.
- **LLM-Powered Reasoning**: Uses **Groq API** (e.g., Llama-3.3-70b) to provide a deep-dive analysis.
- **Fast Inference**: Leverages Groq's LPU for near-instant results.
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

3.  **Configuration:**
    - Get a free API Key from [console.groq.com](https://console.groq.com).
    - Provide it in the sidebar or set `GROQ_API_KEY` environment variable.

4.  **Perform Diagnosis:**
    - Enter metrics and stats.
    - Select a model (e.g., **llama-3.3-70b-versatile**).
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
