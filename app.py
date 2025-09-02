import streamlit as st
import joblib
import os

st.set_page_config(page_title="Fake News Classifier", layout="centered")
st.header("Classificação de Fake News")

input_text = st.text_area("Digite a notícia para análise", height=150)

modelo_path = "model.joblib"
vetor_path = "vectorizer.joblib"

if os.path.exists(modelo_path) and os.path.exists(vetor_path):
    model = joblib.load(modelo_path)
    vectorizer = joblib.load(vetor_path)

    if st.button("Classificar"):
        if input_text.strip():
            texto_vetor = vectorizer.transform([input_text])
            pred = model.predict(texto_vetor)[0]
            st.success(f"Previsão: {pred}")
        else:
            st.warning("Digite algum texto para classificar")
else:
    st.error("Arquivos de modelo ou vetor ausentes.")