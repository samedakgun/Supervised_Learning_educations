# model_training.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_data(filepath):
    """
    Veriyi dosyadan yükler.
    Args:
        filepath (str): Veri dosyasının yolu.
    Returns:
        DataFrame: Veri çerçevesi.
    """
    data = pd.read_csv(filepath)
    return data


def train_model(X_train, y_train, max_depth=None):
    """
    Karar Ağacı modelini eğitir.
    Args:
        X_train (DataFrame): Eğitim özellikleri.
        y_train (Series): Eğitim hedef değişkeni.
        max_depth (int): Karar ağacı derinliği. None olduğunda sınırsız.
    Returns:
        model: Eğitilmiş karar ağacı modeli.
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    print("Model başarıyla eğitildi.")
    return model


def main():
    # 1. Veriyi Yükle
    data = load_data('//Decision_Trees/data/telco.csv')

    # 2. Yalnızca Belirli Özellikleri ve Hedef Değişkeni Seç
    features = ["Age", "Number of Dependents", "Tenure in Months",
                "Avg Monthly Long Distance Charges", "Avg Monthly GB Download",
                "Monthly Charge", "Total Revenue"]
    X = data[features]
    y = data['Churn Label']  # Hedef değişken

    # 3. Eğitim ve Test Setlerine Ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Modeli Eğit
    model = train_model(X_train, y_train, max_depth=5)

    # 5. Modeli Değerlendir
    predictions = model.predict(X_test)
    print("Doğruluk Skoru:", accuracy_score(y_test, predictions))
    print("Sınıflandırma Raporu:\n", classification_report(y_test, predictions))

    # 6. Modeli Kaydet
    joblib.dump(model, '//Decision_Trees/models/decision_tree_model.pkl')
    print("Model 'decision_tree_model_limited.pkl' olarak kaydedildi.")


if __name__ == "__main__":
    main()
