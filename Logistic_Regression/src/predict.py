import pickle
from data_preprocessing import load_and_preprocess_data
from sklearn.feature_extraction.text import CountVectorizer


def predict(text):
    # Eğitilmiş modeli yükleme
    with open('//Logistic_Regression/models/model.pkl', 'rb') as file:
        model, vectorizer, label_encoder = pickle.load(file)

    # Yeni veriyi dönüştürme
    text = [text]
    X = vectorizer.transform(text)

    # Tahmin yapma
    prediction = model.predict(X)

    # Tahmini etiket olarak döndürme
    return "Spam" if prediction[0] == 1 else "Not Spam"


if __name__ == "__main__":
    # Örnek tahmin
    sample_text = "Ücretsiz hediyeler kazanmak için hemen tıklayın!"
    print(f"Tahmin: {predict(sample_text)}")
