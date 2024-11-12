# model_training.py

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_data, preprocess_data, split_data


def train_model(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Random Forest modelini eğitir.
    Args:
        X_train (DataFrame): Eğitim özellikleri.
        y_train (Series): Eğitim hedef değişkeni.
        n_estimators (int): Ağaç sayısı.
        max_depth (int): Maksimum derinlik. None ise sınırsız.
        random_state (int): Rastgelelik için sabit değer.
    Returns:
        model: Eğitilmiş Random Forest modeli.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    print("Model başarıyla eğitildi.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Modeli test verisinde değerlendirir.
    Args:
        model: Eğitilmiş model.
        X_test (DataFrame): Test özellikleri.
        y_test (Series): Test hedef değişkeni.
    Returns:
        dict: Modelin doğruluk ve sınıflandırma raporu.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print("Doğruluk Skoru:", accuracy)
    print("Sınıflandırma Raporu:\n", report)

    return {"accuracy": accuracy, "classification_report": report}


def main():
    # Veriyi yükle ve ön işleme adımlarını yap
    filepath = '//Random_Forest/data/bank.csv'
    data = load_data(filepath)
    X, y, _ = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Modeli eğit
    model = train_model(X_train, y_train, n_estimators=100, max_depth=10)

    # Modeli değerlendir
    metrics = evaluate_model(model, X_test, y_test)

    # Modeli kaydet
    model_path = '//Random_Forest/models/random_forest_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model '{model_path}' olarak kaydedildi.")


if __name__ == "__main__":
    main()
