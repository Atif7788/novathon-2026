import streamlit as st
import pickle
import pandas as pd


st.set_page_config(
    page_title="Cafeteria Crowd Predictor",
    page_icon="🍽️",
    layout="centered"
)

 
data = pd.read_csv("cafeteria_data.csv")
model = pickle.load(open("crowd_model.pkl", "rb"))

st.markdown("<h1>🍽️ AI Cafeteria Crowd Predictor</h1>", unsafe_allow_html=True)

st.info("Select time and day to check cafeteria crowd level.")


hour = st.slider("Select Hour", 8, 17)
day = st.slider("Select Day (1=Mon ... 7=Sun)", 1, 7)


if st.button("Predict"):
    prediction = model.predict([[hour, day]])[0]

    st.write(f"**Predicted Crowd:** {int(prediction)} students")

   
    if prediction > 100:
        st.error("High Crowd ⚠️ Avoid now. Try after 1–2 hours.")
    elif prediction > 60:
        st.warning("Medium Crowd ⏳ Moderate waiting time.")
    else:
        st.success("Low Crowd ✅ Best time to visit now!")

   
    st.subheader("📊 Crowd Trend Analysis")
    st.line_chart(data.groupby("Hour")["Crowd"].mean())

   
    st.markdown("### 🤖 AI Explanation")
    st.write(
        "This system uses a machine learning model trained on historical cafeteria data. "
        "As more data is added, the model becomes more accurate in predicting crowd levels."
    )

st.markdown("---")
st.caption("team leader - Atif ali")
st.caption("Developed by CSE (AI & ML) Team | Novathon 2026")