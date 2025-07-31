import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    PrecisionRecallDisplay # Make sure this is imported for PR curve plotting
)
import warnings

warnings.filterwarnings('ignore') # Filter out warnings, especially from scikit-learn

st.set_page_config(layout="wide", page_title="Fraud Detection Dashboard")

st.title("Fraud Detection Model Evaluation Dashboard")
st.markdown("---") # A horizontal line for visual separation

# --- 0. Load Data and Models ---
# @st.cache_resource decorator caches the loaded models/data
# This prevents reloading them every time a user interacts with the app, speeding it up.
@st.cache_resource
def load_resources():
    loaded_models = {}
    # List the names of the models you trained, exactly as they appear in your 'models' dictionary
    model_names = ["Random Forest", "XGBoost", "SVM", "Logistic Regression", "KNN"]

    st.spinner("Loading models...") # Show a spinner while loading
    for name in model_names:
        try:
            # Construct the filename exactly as you saved it
            filename = f'trained_model_{name.replace(" ", "_").replace("/", "_")}.pkl'
            loaded_models[name] = joblib.load(filename)
            # st.success(f"Loaded {name} model.") # Optional: confirmation message
        except FileNotFoundError:
            st.error(f"Model file for '{name}' not found at '{filename}'. "
                     "Please ensure you ran the saving cell in your Jupyter notebook.")
            return None, None, None # Return None to indicate failure

    st.spinner("Loading test data...")
    try:
        X_test_loaded = pd.read_pickle('X_test.pkl')
        y_test_loaded = pd.read_pickle('y_test.pkl')
        # st.success("Loaded test data (X_test, y_test).") # Optional: confirmation message
    except FileNotFoundError:
        st.error("Test data files ('X_test.pkl', 'y_test.pkl') not found. "
                 "Please ensure you ran the saving cell in your Jupyter notebook.")
        return None, None, None # Return None to indicate failure

    return loaded_models, X_test_loaded, y_test_loaded

# Call the loading function
loaded_models, X_test, y_test = load_resources()

# Exit if resources couldn't be loaded to prevent further errors
if loaded_models is None or X_test is None or y_test is None:
    st.stop() # Stop the Streamlit app execution if resources are missing

# --- 1. Re-evaluate Models for Dashboard Display (or load pre-calculated if saved) ---
# It's generally better to re-calculate here for freshness and to ensure consistency
# with loaded models, or you could save evaluation_results_fast directly from notebook.
evaluation_results = []
y_preds_dict = {}
y_probs_dict = {}

st.subheader("Performing predictions for dashboard metrics...")
for name, model in loaded_models.items():
    y_pred = model.predict(X_test)
    # Check if the model has predict_proba (e.g., SVC needs probability=True)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = [0.5] * len(y_test) # Default or placeholder if no probabilities
        roc_auc = 0.5 # Default ROC AUC if no probabilities

    y_preds_dict[name] = y_pred
    y_probs_dict[name] = y_prob

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0), # Handle cases where no positive predictions
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc
    }
    evaluation_results.append(metrics)

results_df = pd.DataFrame(evaluation_results).round(4)
results_df_sorted = results_df.sort_values(by="ROC AUC", ascending=False).reset_index(drop=True)

st.success("Model evaluation complete!")
st.markdown("---")

# --- 2. Model Performance Summary Table ---
st.header("1. Model Performance Summary")
st.dataframe(results_df_sorted, use_container_width=True) # Makes DataFrame responsive

st.markdown("---")

# --- 3. Interactive Model Selection for Visualizations ---
st.header("2. Detailed Model Visualizations")
selected_models = st.multiselect(
    "**Select models to visualize:**",
    options=list(loaded_models.keys()),
    default=list(loaded_models.keys()) # All models selected by default
)

if not selected_models:
    st.info("Please select at least one model to display visualizations below.")
else:
    # --- 3.1. ROC Curve Plot ---
    st.subheader("2.1. Receiver Operating Characteristic (ROC) Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    for name in selected_models:
        if name in y_probs_dict and hasattr(loaded_models[name], 'predict_proba'):
            fpr, tpr, _ = roc_curve(y_test, y_probs_dict[name])
            roc_auc = roc_auc_score(y_test, y_probs_dict[name])
            ax_roc.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
        else:
            st.warning(f"ROC curve cannot be plotted for {name} as it lacks 'predict_proba'.")

    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.50)')
    ax_roc.set_xlabel('False Positive Rate (FPR)')
    ax_roc.set_ylabel('True Positive Rate (TPR)')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True)
    st.pyplot(fig_roc) # Display the plot in Streamlit

    st.markdown("---")

    # --- 3.2. Confusion Matrices ---
    st.subheader("2.2. Confusion Matrices")
    # Use Streamlit columns for side-by-side display if many models
    num_cols = min(len(selected_models), 3) # Max 3 columns for better readability
    cols = st.columns(num_cols)
    col_idx = 0
    for name in selected_models:
        with cols[col_idx]:
            st.write(f"**{name}**")
            if name in y_preds_dict:
                cm = confusion_matrix(y_test, y_preds_dict[name])
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4)) # Smaller size for column layout
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=['Predicted Negative', 'Predicted Positive'],
                            yticklabels=['Actual Negative', 'Actual Positive'], ax=ax_cm)
                ax_cm.set_xlabel('Predicted Label')
                ax_cm.set_ylabel('True Label')
                ax_cm.set_title(f'Confusion Matrix')
                st.pyplot(fig_cm)
            else:
                st.write("Prediction data not available for this model.")
        col_idx = (col_idx + 1) % num_cols # Move to next column

    st.markdown("---")

    # --- 3.3. Precision-Recall Curves ---
    st.subheader("2.3. Precision-Recall Curve")
    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
    for name in selected_models:
        if name in loaded_models and hasattr(loaded_models[name], 'predict_proba'):
            PrecisionRecallDisplay.from_estimator(loaded_models[name], X_test, y_test, name=name, ax=ax_pr, plot_chance_level=True)
        else:
            st.warning(f"Precision-Recall Curve cannot be plotted for {name} (no 'predict_proba').")
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.grid(True)
    st.pyplot(fig_pr)

    st.markdown("---")

    # --- 3.4. Bar Plot of Key Evaluation Metrics ---
    st.subheader("2.4. Comparison of Key Evaluation Metrics")
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    results_df_selected = results_df_sorted[results_df_sorted['Model'].isin(selected_models)]
    results_df_melted = results_df_selected.melt(id_vars=["Model"], value_vars=metrics_to_plot,
                                             var_name="Metric", value_name="Score")

    fig_bar, ax_bar = plt.subplots(figsize=(14, 7))
    sns.barplot(x="Metric", y="Score", hue="Model", data=results_df_melted, palette="viridis", ax=ax_bar)
    ax_bar.set_title('Comparison of Model Evaluation Metrics')
    ax_bar.set_ylabel('Score')
    ax_bar.set_xlabel('Metric')
    ax_bar.set_ylim(0, 1) # Metrics are typically between 0 and 1
    ax_bar.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    st.pyplot(fig_bar)

st.markdown("---")

# --- 4. Simple Deployment Solution (Prototype Section) ---
st.header("3. Deployment Solution Prototype: Real-time Prediction")
st.write("This section demonstrates how a trained model could be used to predict fraud for new, unseen transactions.")
st.info("This is a simplified prototype. A full deployment solution would involve APIs, robust error handling, and more comprehensive input validation.")

st.markdown("**Enter New Transaction Features:**")
# You'll need to create input widgets for each feature in your X_test
# For simplicity, let's just create a slider for a few features.
# In a real scenario, you'd make inputs for all 20 features, perhaps grouped.
# For now, let's pick a few arbitrary features to demonstrate.
# Replace 'Feature_0', 'Feature_1', etc. with actual feature names if you have them.

# Example: Creating sliders for a few features (adjust min/max/step based on your data's range)
# You can inspect X_test.describe() in your notebook to get min/max values.
feature_inputs = {}
st.write("*(Note: Input ranges are generic. Adjust based on your actual feature distributions.)*")

# Create two columns for better layout of inputs
input_col1, input_col2 = st.columns(2)

with input_col1:
    feature_inputs['Feature_0'] = st.slider('Feature_0 (e.g., Transaction Amount)', float(X_test['Feature_0'].min()), float(X_test['Feature_0'].max()), float(X_test['Feature_0'].mean()))
    feature_inputs['Feature_1'] = st.slider('Feature_1 (e.g., Transaction Frequency)', float(X_test['Feature_1'].min()), float(X_test['Feature_1'].max()), float(X_test['Feature_1'].mean()))
    feature_inputs['Feature_2'] = st.slider('Feature_2', float(X_test['Feature_2'].min()), float(X_test['Feature_2'].max()), float(X_test['Feature_2'].mean()))

with input_col2:
    feature_inputs['Feature_3'] = st.slider('Feature_3', float(X_test['Feature_3'].min()), float(X_test['Feature_3'].max()), float(X_test['Feature_3'].mean()))
    feature_inputs['Feature_4'] = st.slider('Feature_4', float(X_test['Feature_4'].min()), float(X_test['Feature_4'].max()), float(X_test['Feature_4'].mean()))
    feature_inputs['Feature_5'] = st.slider('Feature_5', float(X_test['Feature_5'].min()), float(X_test['Feature_5'].max()), float(X_test['Feature_5'].mean()))


# Select a model for real-time prediction
model_for_prediction = st.selectbox(
    "**Select a model for real-time fraud prediction:**",
    options=list(loaded_models.keys())
)

if st.button("Predict Fraud"):
    # Create a DataFrame for the new input, ensuring column order matches training data
    input_df = pd.DataFrame([feature_inputs]) # Convert dict to DataFrame row

    # Ensure all 20 features are present, fill missing with 0 or mean of training data
    # This is a crucial step for deployment to handle unseen features or order
    # For simplicity with your synthetic data, we assume the initial 6 are sufficient
    # For a real dataset, you'd need all X.columns
    all_features = X_test.columns.tolist() # Get the original feature names
    for col in all_features:
        if col not in input_df.columns:
            input_df[col] = 0.0 # Or use X_train[col].mean() if appropriate

    input_df = input_df[all_features] # Ensure correct column order

    selected_model = loaded_models[model_for_prediction]
    prediction = selected_model.predict(input_df)[0] # Get the first (and only) prediction
    prediction_proba = selected_model.predict_proba(input_df)[:, 1][0] if hasattr(selected_model, 'predict_proba') else 'N/A'

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"**Potential Fraud Detected!** (Probability: {prediction_proba:.4f})" if prediction_proba != 'N/A' else "**Potential Fraud Detected!**")
    else:
        st.success(f"**Transaction Appears Legitimate.** (Probability of Fraud: {prediction_proba:.4f})" if prediction_proba != 'N/A' else "**Transaction Appears Legitimate.**")

    st.markdown("---")


st.markdown("---")
st.markdown("### Project Documentation & Collaboration")
st.markdown("""
This interactive dashboard serves as a key deliverable for fraud detection results.
It fulfills the requirements for **Comprehensive model evaluation and validation**,
**Create confusion matrices and performance metrics**, **Develop model comparison framework**,
and **Build interactive dashboard for results visualization**.

For **Final project presentation and documentation**, this dashboard can be used
as a live demonstration. A formal report with detailed explanations,
findings, and insights derived from these evaluations (including the confusion matrices)
would complement this application.

The **Deployment-ready application prototype**  is demonstrated by the "Real-time Prediction"
section, showing how a trained model can receive new input and provide a prediction.
Further work on a robust API and full integration would be the next steps for a production deployment.

Remember to maintain good **Collaboration Guidelines**  including version control with Git/GitHub,
documentation , and regular communication.
""")
st.markdown("---")
st.markdown("Developed by Sanya - Model Evaluation & Deployment Specialist")

# Optional: Add a 'Download Report' button later if you generate a PDF report
# st.download_button("Download Full Evaluation Report", data=report_pdf_bytes, file_name="Fraud_Detection_Report.pdf", mime="application/pdf")