import streamlit as st
import shap
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

st.title("🔬 SHAP Analysis")

if "df" in st.session_state:

    df = st.session_state["df"]
    target = st.session_state["target"]

    X = df.drop(columns=[target])
    y = df[target]

    model = LGBMRegressor(n_estimators=500)
    model.fit(X, y)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    fig = plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)