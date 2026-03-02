import streamlit as st
from utils.evaluation import analyze_gap

st.title("📈 Leaderboard Gap Analysis")

cv_score = st.number_input("Your CV RMSE", value=0.0)
public_lb = st.number_input("Public LB RMSE", value=0.0)
private_lb = st.number_input("Private LB RMSE", value=0.0)

if st.button("Analyze Gap"):
    result = analyze_gap(cv_score, public_lb, private_lb)
    st.write(result)

    if result["public_vs_private"] > 0:
        st.warning("⚠️ Possible Overfitting")
    else:
        st.success("✅ Stable Model")

# import streamlit as st
# from utils.preprocessing import build_preprocessor
# from utils.modeling import get_models, train_model
# from utils.eda import detect_feature_types

# st.title("🚀 Baseline Model")

# if "df" in st.session_state:
#     df = st.session_state["df"]

#     target = st.selectbox("Target", df.columns)

#     X = df.drop(columns=[target])
#     y = df[target]

#     numeric, categorical = detect_feature_types(X)

#     preprocessor = build_preprocessor(numeric, categorical)

#     models = get_models()

#     for name, model in models.items():
#         score, std = train_model(model, X, y)
#         st.write(f"{name} RMSE: {score:.4f} ± {std:.4f}")