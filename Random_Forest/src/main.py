import joblib
import numpy as np


def get_user_input(label_encoders):
    """
    Sabit özellikleri ve kategorik dönüşümleri kullanarak sayısal değerlere dönüştürür.
    Args:
        label_encoders (dict): LabelEncoder nesnelerini içeren sözlük.
    Returns:
        list: Sayısal değerlere dönüştürülmüş özellikler.
    """
    # Sabit özellikler tanımlanıyor
    features = {
        "age": 23,  # Yaş
        "job": "admin",  # Meslek
        "marital": "single",  # Medeni Durum
        "education": "primary",  # Eğitim Düzeyi
        "default": "no",  # Kredi Temerrütü Var mı?
        "balance": 20000,  # Bakiye
        "housing": "no",  # Konut Kredisi Var mı?
        "loan": "no",  # Kredi Borcu Var mı?
        "contact": "telephone",  # İletişim Türü
        "day": 20,  # Gün
        "month": "feb",  # Ay
        "duration": 20,  # Son Görüşme Süresi
        "campaign": 5,  # Kampanya Sayısı
        "pdays": 10,  # Son Kampanyadan Bu Yana Günler
        "previous": 5,  # Önceki Kampanyalar
        "poutcome": "unknown"  # Önceki Kampanya Sonucu
    }

    # Kategorik özellikleri sayısal değerlere dönüştürme
    for key, encoder in label_encoders.items():
        if key in features:
            try:
                # Eğer kategori mevcutsa, doğrudan dönüştür
                features[key] = encoder.transform([features[key]])[0]
            except ValueError:
                # Bilinmeyen kategori durumunda 'unknown' veya varsayılan en sık değeri kullan
                most_frequent = encoder.classes_[0] if encoder.classes_.size > 0 else 'unknown'
                if 'unknown' in encoder.classes_:
                    features[key] = encoder.transform(['unknown'])[0]
                else:
                    features[key] = encoder.transform([most_frequent])[0]
                print(f"Bilinmeyen değer '{features[key]}' için '{most_frequent}' veya 'unknown' değeri kullanıldı.")

    return list(features.values())

def main():
    # Modeli ve label encoders'ı yükle
    model_path = '//Random_Forest/models/random_forest_model.pkl'
    model = joblib.load(model_path)
    print(f"Model '{model_path}' dosyasından başarıyla yüklendi.")

    encoder_path = '//Random_Forest/models/label_encoders.pkl'
    label_encoders = joblib.load(encoder_path)
    print(f"Label encoders '{encoder_path}' dosyasından başarıyla yüklendi.")

    # Sabit kullanıcı girdisini al ve sayısal değerlere dönüştür
    user_data = np.array(get_user_input(label_encoders)).reshape(1, -1)

    # Tahmin yap
    prediction = model.predict(user_data)
    result = "Müşteri Ürün Alacak" if prediction[0] == 1 else "Müşteri Ürün Almayacak"

    print("\nTahmin Sonucu:", result)


if __name__ == "__main__":
    main()
