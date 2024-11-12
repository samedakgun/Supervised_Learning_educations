# evaluation.py

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


def evaluate_model(model, X_test, y_test):
    """
    Modelin performansını test verisi üzerinde değerlendirir.
    Args:
        model: Eğitilmiş model.
        X_test (DataFrame): Test özellikleri.
        y_test (Series): Test hedef değişkeni.
    Returns:
        dict: Doğruluk skoru, sınıflandırma raporu ve karmaşıklık matrisi.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)

    print("Modelin doğruluğu:", accuracy)
    print("\nSınıflandırma Raporu:\n", report)
    print("\nKarmaşıklık Matrisi:\n", confusion)

    return {"accuracy": accuracy, "classification_report": report, "confusion_matrix": confusion}


def load_model(filepath):
    """
    Kaydedilmiş modeli dosyadan yükler.
    Args:
        filepath (str): Modelin dosya yolu.
    Returns:
        Model: Yüklü model.
    """
    try:
        model = joblib.load(filepath)
        print(f"Model '{filepath}' dosyasından başarıyla yüklendi.")
        return model
    except FileNotFoundError:
        print("Model dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"Model yüklenirken bir hata oluştu: {e}")


def main():
    # 1. Veriyi Yükleme
    data = load_data('//Decision_Trees/data/telco.csv')

    # 2. Test Setini Hazırlama
    if 'Churn Label' in data.columns:
        y = data['Churn Label']
        X = data.drop(columns=['Churn Label'])
    else:
        print("Hedef değişken 'Churn Label' bulunamadı.")
        return

    # Sadece sayısal sütunları seç
    X = X.select_dtypes(include=['number'])

    # Veriyi yeniden eğitim ve test setlerine ayır
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Modeli Yükleme
    model = load_model('//Decision_Trees/models/decision_tree_model.pkl')

    # Model başarıyla yüklendiyse değerlendir
    if model:
        metrics = evaluate_model(model, X_test, y_test)

        # Test etme: Modelin tahminleriyle manuel olarak doğrulama
        test_predictions = model.predict(X_test[:5])
        print("Modelin test setindeki ilk 5 tahmini:", test_predictions)
        print("Gerçek değerler:", y_test.iloc[:5].values)


if __name__ == "__main__":
    main()
