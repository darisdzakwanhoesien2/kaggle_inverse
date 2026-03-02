import streamlit as st
from utils.preprocessing import build_preprocessor
from utils.modeling import get_models, evaluate_model
from utils.eda import detect_feature_types

st.title("🚀 Baseline Models")

if "df" in st.session_state:

    df = st.session_state["df"]
    target = st.session_state["target"]

    X = df.drop(columns=[target])
    y = df[target]

    numeric, categorical = detect_feature_types(X)

    models = get_models()

    results = {}

    for name, model in models.items():
        score, std = evaluate_model(model, X, y)
        results[name] = (score, std)
        st.write(f"{name}: RMSE {score:.4f} ± {std:.4f}")

    st.session_state["baseline_results"] = results