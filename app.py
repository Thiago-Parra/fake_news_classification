import streamlit as st
import joblib
import os

st.set_page_config(page_title="Fake News Classifier", layout="centered")
st.header("Fake News Classifier")

input_text = st.text_area("Enter the news title for analysis", height=150)

modelo_path = "model.joblib"
vetor_path = "vectorizer.joblib"

if os.path.exists(modelo_path) and os.path.exists(vetor_path):
    model = joblib.load(modelo_path)
    vectorizer = joblib.load(vetor_path)

    if st.button("Classify"):
        if input_text.strip():
            texto_vetor = vectorizer.transform([input_text])
            pred = model.predict(texto_vetor)[0]
            st.success(f"Prediction: {pred}")
        else:
            st.warning("Please enter some text to classify.")
else:
    st.error("Model or vectorizer files are missing.")