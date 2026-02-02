import streamlit as st
import pandas as pd
import numpy as np
from logic import calculate_rules, diagnose_failure
from drift_logic import calculate_drift_stats, explain_drift
from health_logic import check_quality, calculate_health_score, generate_health_summary
import os

# Page Config
st.set_page_config(
    page_title="ML Model Failure Diagnosis Engine",
    page_icon="🩺",
    layout="wide"
)

# Title and Description
st.title("🩺 ML Model Failure Diagnosis Engine")

# Sidebar - API Key Configuration
with st.sidebar:
    st.header("Configuration")
    api_key_env = os.getenv("GROQ_API_KEY")
    api_key_input = st.text_input(
        "Groq API Key",
        value=api_key_env if api_key_env else "",
        type="password",
        help="Enter your Groq API Key here. Get one for free at console.groq.com."
    )
    st.info("The API key is used for LLM analysis.")

    model_name = st.selectbox(
        "Select Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0,
        help="Select the model to use. 'llama-3.3-70b' is the latest and most capable."
    )

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Model Diagnosis", "📊 Dataset Shift & Drift", "🏥 Dataset Health"])

# --- TAB 1: MODEL FAILURES ---
with tab1:
    st.markdown("""
    **Diagnose why your ML model is underperforming.**
    Enter your training metrics and dataset statistics to get a diagnosis.
    """)
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Model Metrics")
        train_acc = st.slider("Training Accuracy", 0.0, 1.0, 0.85, 0.01)
        val_acc = st.slider("Validation Accuracy", 0.0, 1.0, 0.70, 0.01)
        train_loss = st.number_input("Training Loss", value=0.15, format="%.4f")
        val_loss = st.number_input("Validation Loss", value=0.45, format="%.4f")

    with col2:
        st.subheader("2. Dataset Statistics")
        num_samples = st.number_input("Number of Samples", min_value=1, value=1000)
        num_features = st.number_input("Number of Features", min_value=1, value=20)
        num_classes = st.number_input("Number of Classes", min_value=2, value=2)
        
        st.caption("Enter comma-separated percentages (e.g., 90, 10)")
        class_dist_str = st.text_input("Class Percentages", value="50, 50")

    # Process Class Distribution
    class_dist = {}
    try:
        if class_dist_str:
            parts = [float(x.strip()) for x in class_dist_str.split(',')]
            for i, p in enumerate(parts):
                class_dist[f"Class {i}"] = p
    except ValueError:
        st.error("Invalid Class Percentages format.")

    metrics = {
        "train_acc": train_acc, "val_acc": val_acc,
        "train_loss": train_loss, "val_loss": val_loss
    }
    stats = {
        "num_samples": num_samples, "num_features": num_features,
        "num_classes": num_classes, "class_dist": class_dist
    }

    st.divider()

    if st.button("Run Model Diagnosis", type="primary"):
        with st.status("Analyzing...", expanded=True) as status:
            st.write("Computing signals...")
            rules = calculate_rules(metrics, stats)
            st.write("Consulting LLM...")
            diagnosis = diagnose_failure(metrics, stats, rules, api_key_input, model_name)
            status.update(label="Complete!", state="complete", expanded=False)

        if rules:
            with st.expander("🚨 Rule-based Signals", expanded=True):
                for rule in rules:
                    st.warning(rule)
        
        st.markdown("### 🧠 AI Analysis")
        st.markdown(diagnosis)

# --- HELPER: LOAD DEMO DATA ---
@st.cache_data
def load_data_from_url(url):
    return pd.read_csv(url)

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
IRIS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# --- TAB 2: DATASET DRIFT ---
with tab2:
    st.markdown("Compare your **Reference (Training)** dataset with your **Current (Production)** dataset to detect drift.")
    
    # Demo Mode Toggle
    use_demo = st.checkbox("🔍 Use Demo Data", help="Load online datasets to see the tool in action without uploading files.")

    c1, c2 = st.columns(2)
    
    if use_demo:
        st.info("Using **Titanic** dataset: Reference (First 500 rows) vs Current (Next rows). fetched from online.")
        # Load Titanic once
        try:
            full_df = load_data_from_url(TITANIC_URL)
            ref_df = full_df.iloc[:500]
            curr_df = full_df.iloc[500:]
        except Exception as e:
            st.error(f"Failed to load demo data: {e}")
            ref_df = None
            curr_df = None
    else:
        with c1:
            ref_file = st.file_uploader("Upload Reference Dataset (CSV)", type="csv")
            if ref_file: ref_df = pd.read_csv(ref_file)
        with c2:
            curr_file = st.file_uploader("Upload Current Dataset (CSV)", type="csv")
            if curr_file: curr_df = pd.read_csv(curr_file)
        
    if 'ref_df' in locals() and 'curr_df' in locals() and ref_df is not None and curr_df is not None:
        try:
            # (Rest of the drift logic uses ref_df and curr_df directly)
            
            st.subheader("1. Data Overview")
            d1, d2 = st.columns(2)
            d1.metric("Reference Rows", ref_df.shape[0], f"{ref_df.shape[1]} cols")
            d2.metric("Current Rows", curr_df.shape[0], f"{curr_df.shape[1]} cols")
            
            # Simple Column Selector for Viz
            st.subheader("2. Compare Features")
            common_cols = list(set(ref_df.columns) & set(curr_df.columns))
            selected_col = st.selectbox("Select Feature to Visualize", common_cols)
            
            if selected_col:
                # Viz based on type
                if pd.api.types.is_numeric_dtype(ref_df[selected_col]):
                    chart_data = pd.DataFrame({
                        "Reference": ref_df[selected_col],
                        "Current": curr_df[selected_col]
                    })
                    st.line_chart(chart_data) # Simple line chart or could use st.bar_chart for histograms if binned
                    # Better: histogram
                    # st.bar_chart is not ideal for overlapping histograms without preprocessing.
                    # We'll stick to a simple description for now or use Altair if needed.
                    # Let's just show stats side by side.
                    s1 = ref_df[selected_col].describe()
                    s2 = curr_df[selected_col].describe()
                    st.dataframe(pd.DataFrame({"Ref": s1, "Curr": s2}).T)
                else:
                    # Categorical Bar Chart
                    v1 = ref_df[selected_col].value_counts(normalize=True)
                    v2 = curr_df[selected_col].value_counts(normalize=True)
                    df_cat = pd.DataFrame({"Ref": v1, "Curr": v2})
                    st.bar_chart(df_cat)

            st.divider()

            if st.button("Calculate Drift & Explain", type="primary"):
                with st.spinner("Calculating Drift Statistics..."):
                    report = calculate_drift_stats(ref_df, curr_df)
                
                # Show Drifting Features
                st.subheader("3. Drift Detection Results")
                
                # Numerical Table
                if report["numerical_drift"]:
                    drift_df = pd.DataFrame(report["numerical_drift"])
                    # Highlight rows where drift_detected is True
                    st.write("Numerical Features:")
                    st.dataframe(
                        drift_df.style.apply(lambda x: ['background-color: #ffcdd2' if x.drift_detected else '' for i in x], axis=1),
                        use_container_width=True
                    )
                
                # Categorical Table
                if report["categorical_drift"]:
                    cat_drift_df = pd.DataFrame(report["categorical_drift"])
                    st.write("Categorical Features:")
                    st.dataframe(
                        cat_drift_df.style.apply(lambda x: ['background-color: #ffcdd2' if x.drift_detected else '' for i in x], axis=1),
                        use_container_width=True
                    )

                # LLM Explanation
                with st.spinner("Generating Explanations with LLM..."):
                    explanation = explain_drift(report, api_key_input, model_name)
                
                st.subheader("🧠 AI Drift Explanation")
                st.markdown(explanation)

        except Exception as e:
            st.error(f"Error processing files: {e}")

# --- TAB 3: DATASET HEALTH ---
with tab3:
    st.markdown("Assess the **readiness and quality** of your dataset for ML training.")
    
    # Demo Mode Toggle for Health
    use_health_demo = st.checkbox("🔍 Use Demo Data (Health Check)", help="Load online datasets to see the health check in action.")

    df_health = None

    if use_health_demo:
        demo_choice = st.selectbox("Select Demo Dataset", ["Titanic (Messy)", "Iris (Clean)"])
        
        try:
            if "Titanic" in demo_choice:
                st.info("Fetching **Titanic** dataset from GitHub...")
                df_health = load_data_from_url(TITANIC_URL)
            else:
                st.info("Fetching **Iris** dataset from GitHub...")
                df_health = load_data_from_url(IRIS_URL)
        except Exception as e:
            st.error(f"Failed to load demo data: {e}")

    else:
        health_file = st.file_uploader("Upload Dataset (CSV)", type="csv", key="health_upload")
        if health_file:
            df_health = pd.read_csv(health_file)
        
    if df_health is not None:
        try:
            # 1. Dataset Overview
            st.subheader("1. Dataset Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", df_health.shape[0])
            c2.metric("Columns", df_health.shape[1])
            c3.metric("Numerical", len(df_health.select_dtypes(include=[np.number]).columns))
            c4.metric("Categorical", len(df_health.select_dtypes(exclude=[np.number]).columns))
            
            with st.expander("👀 View Raw Data (First 5 Rows)"):
                st.dataframe(df_health.head())
            
            st.divider()
            
            # 2. Run Health Check
            if st.button("Run Health Check", type="primary"):
                with st.spinner("Analyzing Data Quality..."):
                    quality_report = check_quality(df_health)
                    score, status = calculate_health_score(quality_report)
                
                # Display Score
                st.subheader(f"Health Score: {score}/100")
                if "Healthy" in status:
                    st.success(f"Status: {status}")
                elif "Needs Attention" in status:
                    st.warning(f"Status: {status}")
                else:
                    st.error(f"Status: {status}")
                    
                # Display Details
                st.subheader("2. Quality Check Results")
                
                with st.expander("Missing Values", expanded=True):
                    if quality_report["missing"]:
                        st.info("**Missing Status**: ✅ = Low Risk (<10%), ❌ = High Risk (>10%).")
                        
                        miss_df = pd.DataFrame(quality_report["missing"]).T
                        miss_df = miss_df.rename(columns={"pct": "Missing %", "count": "Missing Count", "flagged": "Missing Status"})
                        miss_df["Missing Status"] = miss_df["Missing Status"].map({True: "❌", False: "✅"})
                        
                        st.dataframe(miss_df.style.format({"Missing %": "{:.1%}", "Missing Count": "{:.0f}"}))
                    else:
                        st.success("No missing values detected.")
                        
                with st.expander("Outliers & Stability", expanded=True):
                    if quality_report["outliers"]:
                        st.info("""
                        **Metric Explanations:**
                        - **Outlier %**: Percentage of rows classified as outliers (Interquartile Range method).
                        - **Outlier Status**: ✅ = Low Risk (<5%), ❌ = High Risk (>5%).
                        - **Stability Status**: ✅ = Healthy (Has Variance), ❌ = Constant (Zero Variance).
                        """)
                        
                        out_df = pd.DataFrame(quality_report["outliers"]).T
                        # Add stability info
                        stab_df = pd.DataFrame(quality_report["stability"]).T
                        
                        combined = out_df.join(stab_df, lsuffix="_out", rsuffix="_stab")
                        
                        # Calculate explicit Variance
                        combined["Variance"] = combined["std"] ** 2
                        
                        # Rename for clarity
                        combined = combined.rename(columns={
                            "pct": "Outlier %",
                            "count": "Outlier Count",
                            "flagged_out": "Outlier Status", # Was High Outlier Risk?
                            "std": "Std Dev",
                            "flagged_stab": "Stability Status" # Was Is Constant?
                        })
                        
                        combined["Outlier Status"] = combined["Outlier Status"].map({True: "❌", False: "✅"})
                        combined["Stability Status"] = combined["Stability Status"].map({True: "❌", False: "✅"})
                        
                        # Select relevant columns to display
                        cols_to_show = ["Outlier %", "Outlier Count", "Outlier Status", "mean", "Std Dev", "Variance", "min", "max", "Stability Status"]
                        # Filter only existing columns just in case
                        cols_to_show = [c for c in cols_to_show if c in combined.columns]
                        
                        st.dataframe(combined[cols_to_show].style.format({
                            "Outlier %": "{:.1%}", 
                            "mean": "{:.2f}", 
                            "Std Dev": "{:.2f}",
                            "Variance": "{:.2f}"
                        }))
                    else:
                        st.info("No numerical features to analyze.")

                with st.expander("Class Imbalance & Cardinality"):
                     if quality_report["imbalance"]:
                        st.info("""
                        **Metric Explanations:**
                        - **Top Class %**: The percentage of the dataset that belongs to the most frequent category. 
                          - *Example: 90% means 9 out of 10 rows are the same class (High Imbalance).*
                        - **Unique Values**: The number of distinct categories found.
                        - **High Risk?**: ✅ = Yes (Risk Detected), ❌ = No (Safe).
                        """)
                        
                        imb_df = pd.DataFrame(quality_report["imbalance"]).T
                        # Rename for clarity
                        imb_df = imb_df.rename(columns={"top_pct": "Top Class %", "flagged": "High Risk?"})
                        imb_df["High Risk?"] = imb_df["High Risk?"].map({True: "✅", False: "❌"})
                        
                        card_df = pd.DataFrame(quality_report["cardinality"]).T
                        card_df = card_df.rename(columns={"count": "Unique Values", "flagged": "High Risk?"})
                        card_df["High Risk?"] = card_df["High Risk?"].map({True: "✅", False: "❌"})
                        
                        c_imb, c_card = st.columns(2)
                        with c_imb:
                            st.write("**Imbalance**")
                            st.dataframe(imb_df.style.format({"Top Class %": "{:.1%}"}))
                        with c_card:
                            st.write("**Cardinality**")
                            st.dataframe(card_df)
                     else:
                        st.info("No categorical features to analyze.")

                # 3. LLM Summary
                with st.spinner("Generating Health Summary..."):
                    health_summary = generate_health_summary(quality_report, score, status, api_key_input, model_name)
                
                st.subheader("🧠 AI Health Summary")
                st.markdown(health_summary)
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
