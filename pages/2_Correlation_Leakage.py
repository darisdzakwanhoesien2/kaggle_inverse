import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.title("🔎 Correlation & Leakage")

if "df" in st.session_state:
    df = st.session_state["df"]
    target = st.session_state["target"]

    numeric_df = df.select_dtypes(include="number")

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    if target in corr.columns:
        st.subheader("Correlation with Target")
        st.write(corr[target].sort_values(ascending=False))