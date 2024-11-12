from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


def evaluate_model(model, X_test, y_test):
    """
    Modeli test verisinde değerlendirir ve metrikleri döndürür.
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
    confusion = confusion_matrix(y_test, predictions)

    print("Doğruluk Skoru:", accuracy)
    print("Sınıflandırma Raporu:\n", report)
    print("Karmaşıklık Matrisi:\n", confusion)

    # Karmaşıklık matrisini görselleştirme
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title("Karmaşıklık Matrisi")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.show()

    return {"accuracy": accuracy, "classification_report": report, "confusion_matrix": confusion}


def main():
    # Kaydedilmiş modeli yükle
    model_path = '//Random_Forest/models/random_forest_model.pkl'
    model = joblib.load(model_path)
    print(f"Model '{model_path}' dosyasından başarıyla yüklendi.")

    # Test verisini yükle
    from data_preprocessing import load_data, preprocess_data, split_data
    data = load_data('//Random_Forest/data/bank.csv')
    X, y, _ = preprocess_data(data)
    _, X_test, _, y_test = split_data(X, y)

    # Modeli değerlendir
    metrics = evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
