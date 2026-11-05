
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification")
st.write(
    "Machine Learning Assignment 2 - Comparison of "
    "five classification algorithms using the Dry Bean dataset."
)

# --------------------------------------------------
# 2. LOAD SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            "models/logistic_regression.pkl"
        ),
        "Decision Tree": joblib.load(
            "models/decision_tree.pkl"
        ),
        "KNN": joblib.load(
            "models/knn.pkl"
        ),
        "Gaussian Naive Bayes": joblib.load(
            "models/gaussian_naive_bayes.pkl"
        ),
        "Support Vector Machine": joblib.load(
            "models/svm.pkl"
        )
    }

    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    return models, scaler, label_encoder


models, scaler, label_encoder = load_models()

# --------------------------------------------------
# 3. MODEL SELECTION
# --------------------------------------------------

st.subheader("Select Classification Model")

selected_model_name = st.selectbox(
    "Choose a Machine Learning Algorithm",
    list(models.keys())
)

selected_model = models[selected_model_name]

# --------------------------------------------------
# 4. CSV FILE UPLOAD
# --------------------------------------------------

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing 16 features and the Class column",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("### Uploaded Dataset Preview")
    st.dataframe(data.head())

    # --------------------------------------------------
    # 5. DATA VALIDATION
    # --------------------------------------------------

    target_column = "Class"

    if target_column not in data.columns:
        st.error("The uploaded CSV must contain a 'Class' column.")

    else:

        feature_columns = [
            "Area", "Perimeter", "MajorAxisLength",
            "MinorAxisLength", "AspectRatio", "Eccentricity",
            "ConvexArea", "EquivDiameter", "Extent",
            "Solidity", "Roundness", "Compactness",
            "ShapeFactor1", "ShapeFactor2",
            "ShapeFactor3", "ShapeFactor4"
        ]

        missing_columns = [
            col for col in feature_columns
            if col not in data.columns
        ]

        if missing_columns:
            st.error(f"Missing feature columns: {missing_columns}")

        else:

            X = data[feature_columns].copy()
            y_actual = data[target_column]

            if X.isnull().any().any() or y_actual.isnull().any():
                st.error("The uploaded dataset contains missing values.")

            else:

                # --------------------------------------------------
                # 6. PREPROCESSING
                # --------------------------------------------------

                if selected_model_name in [
                    "Logistic Regression", "KNN",
                    "Support Vector Machine"
                ]:
                    X_processed = scaler.transform(X)

                else:
                    X_processed = X

                # --------------------------------------------------
                # 7. PREDICTIONS
                # --------------------------------------------------

                y_pred_encoded = selected_model.predict(X_processed)

                y_pred = label_encoder.inverse_transform(
                    y_pred_encoded.astype(int)
                )

                y_actual_encoded = label_encoder.transform(y_actual)

                # --------------------------------------------------
                # 8. MODEL EVALUATION
                # --------------------------------------------------

                y_prob = selected_model.predict_proba(X_processed)

                accuracy = accuracy_score(y_actual_encoded, y_pred_encoded)

                auc = roc_auc_score(
                    y_actual_encoded,
                    y_prob,
                    multi_class="ovr",
                    average="weighted"
                )

                precision = precision_score(
                    y_actual_encoded,
                    y_pred_encoded,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_actual_encoded,
                    y_pred_encoded,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_actual_encoded,
                    y_pred_encoded,
                    average="weighted",
                    zero_division=0
                )

                mcc = matthews_corrcoef(
                    y_actual_encoded,
                    y_pred_encoded
                )

                # --------------------------------------------------
                # 9. DISPLAY METRICS
                # --------------------------------------------------

                st.subheader("Model Performance Metrics")

                col1, col2, col3 = st.columns(3)

                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("AUC", f"{auc:.4f}")
                col3.metric("Precision", f"{precision:.4f}")

                col4, col5, col6 = st.columns(3)

                col4.metric("Recall", f"{recall:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC", f"{mcc:.4f}")

                # --------------------------------------------------
                # 10. CONFUSION MATRIX
                # --------------------------------------------------

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_actual_encoded,
                    y_pred_encoded
                )

                fig, ax = plt.subplots(figsize=(9, 6))

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_,
                    ax=ax
                )

                ax.set_xlabel("Predicted Class")
                ax.set_ylabel("Actual Class")
                ax.set_title(selected_model_name)

                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                # --------------------------------------------------
                # 11. CLASSIFICATION REPORT
                # --------------------------------------------------

                st.subheader("Classification Report")

                report = classification_report(
                    y_actual_encoded,
                    y_pred_encoded,
                    target_names=label_encoder.classes_,
                    output_dict=True,
                    zero_division=0
                )

                st.dataframe(
                    pd.DataFrame(report).transpose().round(4)
                )

                # --------------------------------------------------
                # 12. PREDICTION RESULTS
                # --------------------------------------------------

                st.subheader("Prediction Results")

                results = data.copy()
                results["Predicted Class"] = y_pred

                st.dataframe(results)

                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="dry_bean_predictions.csv",
                    mime="text/csv"
                )

else:
    st.info("Please upload your test_data.csv file to evaluate the model.")
