import streamlit as st
from utils.preprocessing import (
    build_preprocessor,
    detect_feature_types,
    convert_datetime_features
)
from utils.modeling import get_models, evaluate_model

st.title("🚀 Baseline Models")

if "df" in st.session_state:

    df = st.session_state["df"].copy()
    target = st.session_state["target"]

    X = df.drop(columns=[target])
    y = df[target]

    # Detect datetime columns
    numeric, categorical, datetime_cols = detect_feature_types(X)

    # Convert datetime to numeric features
    if datetime_cols:
        X = convert_datetime_features(X, datetime_cols)
        numeric, categorical, _ = detect_feature_types(X)

    preprocessor = build_preprocessor(numeric, categorical)

    models = get_models()

    for name, model in models.items():
        score, std = evaluate_model(model, preprocessor, X, y)
        st.write(f"{name}: RMSE {score:.4f} ± {std:.4f}")

# import streamlit as st
# from utils.preprocessing import build_preprocessor
# from utils.modeling import get_models, evaluate_model
# from utils.eda import detect_feature_types

# st.title("🚀 Baseline Models")

# if "df" in st.session_state:

#     df = st.session_state["df"]
#     target = st.session_state["target"]

#     X = df.drop(columns=[target])
#     y = df[target]

#     numeric, categorical = detect_feature_types(X)

#     models = get_models()

#     results = {}

#     for name, model in models.items():
#         score, std = evaluate_model(model, X, y)
#         results[name] = (score, std)
#         st.write(f"{name}: RMSE {score:.4f} ± {std:.4f}")

#     st.session_state["baseline_results"] = results