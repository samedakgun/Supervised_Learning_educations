from data_preprocessing import load_data, preprocess_data, split_data
from model_training import train_svm, save_model
from evaluation import evaluate_model, print_metrics
import numpy as np
import joblib


def main():
    # 1. Veri Yükleme
    data_path = '//SVM/data/data.csv'
    data = load_data(data_path)

    # 2. Veri Ön İşleme
    X, y = preprocess_data(data)
    if X is None or y is None:
        print("Veri setinde sorun var, işlem sonlandırıldı.")
        return

    # 3. Veriyi Eğitim ve Test Setlerine Ayırma
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Model Eğitimi
    model = train_svm(X_train, y_train, kernel='linear')

    # 5. Modeli Kaydetme
    model_path = '//SVM/models/svm_model.pkl'
    save_model(model, model_path)

    # 6. Modeli Test Etme ve Değerlendirme
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)

    # 7. Örnek Veri ile Test Etme
    # Yeni bir örnek veri oluşturalım (bu veriyi eğitim setindeki özelliklere göre manuel olarak girmelisiniz)
    # Aşağıdaki örnek rastgele sayılardan oluşuyor ve her bir özellik için uygun aralıkta olması gerekiyor.
    new_sample = np.array([[14.0, 20.0, 90.0, 500.0, 0.1, 0.2, 0.15, 0.1, 0.2, 0.07,
                            0.4, 1.0, 3.0, 30.0, 0.007, 0.03, 0.04, 0.01, 0.02, 0.004,
                            15.0, 25.0, 120.0, 800.0, 0.14, 0.3, 0.4, 0.2, 0.3, 0.08]])

    # Modelin tahminini alın
    prediction = model.predict(new_sample)

    # Tahmini sonuç
    print("Yeni örnek için tahmin edilen sınıf:",
          "Malignant (Kötü Huylu)" if prediction[0] == 1 else "Benign (İyi Huylu)")


if __name__ == "__main__":
    main()
