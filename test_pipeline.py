import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

model_path = "model.joblib"
vectorizer_path = "vectorizer.joblib"

# Lista de cenários de teste: (texto, classe_esperada)
test_cases = [
    # Fake news
    ("Donald Trump Sends Out Unhinged Letter To Media Demanding They Retract Every Story About His Russia Ties", "fake"),
    ("JUST IN: President Trump’s First Official Act Was To Give The CIA The Power To Keep Secrets From The American People", "fake"),
    # Real news
    ("House Republicans Fret About Winning Their Health Care Suit", "real"),
    ("The U.S. has a new strategy for Afghanistan. Will it work?", "real"),
]

def test_model_files_exist():
    assert os.path.exists("./model.joblib"), "Modelo não encontrado"
    assert os.path.exists("./vectorizer.joblib"), "Vetor não encontrado"

def test_vectorizer_output_shape():
    vectorizer = joblib.load("./vectorizer.joblib")
    sample = ["This is a sample title."]
    vetor = vectorizer.transform(sample)
    assert vetor.shape[0] == 1, "Vetor retornou de forma incorreta"

def test_model_prediction_labels():
    model = joblib.load("./model.joblib")
    vectorizer = joblib.load("./vectorizer.joblib")
    sample = ["This is a sample title."]
    vetor = vectorizer.transform(sample)
    pred = model.predict(vetor)[0]
    assert pred in ["fake", "real"], f"Rótulo inesperado {pred}"

def test_data_validation():
    df = pd.read_csv("./data/news_limpo.csv")
    assert "title" in df.columns and "label" in df.columns
    assert df["title"].notnull().all()
    assert df["label"].isin(["fake","real"]).all()

def test_fake_news_classification():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    for title, expected in test_cases:
        pred = model.predict(vectorizer.transform([title]))[0]
        print(f"Título: {title}\nPrevisto: {pred} | Esperado: {expected}\n")
        assert pred == expected
