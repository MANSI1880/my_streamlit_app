import streamlit as st
import pandas as pd

from backend import make_prediction


st.set_page_config(page_title="Construction Cost Prediction")

st.title("🏗️ Linear Regression — Construction Cost Prediction")


# -------------------------
# Upload dataset just to read columns
# -------------------------
file = st.file_uploader("Upload your dataset (same structure as training data)", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Preview:", df.head())

    target = "Total_Estimate"   # <<< yaha bhi apna target column name daalo
    feature_cols = [c for c in df.columns if c != target]

    st.write("### Enter values for prediction")

    user_input = {}

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            val = float(df[col].median())
            user_input[col] = st.number_input(col, value=val)
        else:
            opts = df[col].dropna().unique().tolist()
            if len(opts) == 0:
                user_input[col] = ""
            else:
                user_input[col] = st.selectbox(col, opts)

    if st.button("Predict Cost"):
        result = make_prediction(user_input)
        st.success(f"Predicted {target}: {result:,.2f}")
else:
    st.info("Upload a CSV file to start.")
