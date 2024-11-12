import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data


def evaluate_model():
    # Veriyi yükleme ve ön işleme
    X, y, vectorizer, label_encoder = load_and_preprocess_data('//Logistic_Regression/data/emails.csv')

    # Eğitim ve test verilerini ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Eğitilmiş modeli yükleme
    with open('//Logistic_Regression/models/model.pkl', 'rb') as file:
        model, vectorizer, label_encoder = pickle.load(file)

    # Test verisi üzerinde tahmin yapma
    y_pred = model.predict(X_test)

    # Performans metriklerini hesaplama
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Sonuçları yazdırma
    print("Model Performansı:")
    print(f"Doğruluk (Accuracy): {accuracy}")
    print(f"Hassasiyet (Precision): {precision}")
    print(f"Duyarlılık (Recall): {recall}")
    print(f"F1 Skoru: {f1}")


if __name__ == "__main__":
    evaluate_model()
