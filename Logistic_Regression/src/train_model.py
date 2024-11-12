import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data


def train_model():
    # Veriyi yükleme ve ön işleme
    X, y, vectorizer, label_encoder = load_and_preprocess_data('//Logistic_Regression/data/emails.csvv')

    # Eğitim ve test verilerini ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Lojistik Regresyon modelini eğitme
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Modeli ve vektörizeri kaydetme
    with open('//Logistic_Regression/models/model.pkl', 'wb') as file:
        pickle.dump((model, vectorizer, label_encoder), file)

    print("Model eğitildi ve models/model.pkl olarak kaydedildi.")


if __name__ == "__main__":
    train_model()
