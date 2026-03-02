import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data_loader import load_data
from utils.eda import analyze_target, missing_report

st.title("📊 Data Profiling")

uploaded_file = st.file_uploader("Upload CSV")

if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state["df"] = df
    st.write(df.head())

    target = st.selectbox("Select Target Column", df.columns)
    st.session_state["target"] = target

    stats = analyze_target(df, target)
    st.write("Target Stats:", stats)

    fig, ax = plt.subplots()
    sns.histplot(df[target], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Missing Value Ratio")
    st.dataframe(missing_report(df))