import streamlit as st
from utils.stacking import build_stacking_model, evaluate_stacking

st.title("🏆 Model Comparison + Stacking")

if "df" in st.session_state:

    df = st.session_state["df"]
    target = st.session_state["target"]

    X = df.drop(columns=[target])
    y = df[target]

    stack_model = build_stacking_model()
    score, std = evaluate_stacking(stack_model, X, y)

    st.write(f"🔥 Stacking RMSE: {score:.4f} ± {std:.4f}")