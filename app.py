import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load your trained model
model = load("/home/vigneshbhatn/Vignesh/Projects/Telecom Churn Project/telecom-churn-prediction/Model/random_forest_churn_model.pkl")

st.title("Bulk Churn Prediction")
st.write("Upload a CSV file with customer data to get churn predictions.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Read file
    df = pd.read_csv(uploaded_file)

    # Display original data
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Preprocess just like training
    df_processed = df.copy()
    if "Churn" in df_processed.columns:
        df_processed.drop(columns=["Churn"], inplace=True)
    if "State" in df_processed.columns:
        df_processed.drop(columns=["State"], inplace=True)

    # Map Yes/No to 1/0
    df_processed["International plan"] = df_processed["International plan"].map({"Yes": 1, "No": 0})
    df_processed["Voice mail plan"] = df_processed["Voice mail plan"].map({"Yes": 1, "No": 0})

    # Reorder columns if needed (must match training data exactly!)
    expected_cols = model.feature_names_in_  # only if your model has this attribute
    df_processed = df_processed[expected_cols]

    # Predict
    preds = model.predict(df_processed)

    # Output
    df["Predicted Churn"] = preds
    st.subheader("Predictions")
    st.dataframe(df)

    # Download result
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", csv, "churn_predictions.csv", "text/csv")
