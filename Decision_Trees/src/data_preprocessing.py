# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    """
    Veri setini yükler.
    Args:
        filepath (str): Veri setinin dosya yolu.
    Returns:
        DataFrame: Yüklenen veri seti.
    """
    try:
        data = pd.read_csv(filepath)
        print("Veri başarıyla yüklendi.")
        return data
    except FileNotFoundError:
        print("Dosya bulunamadı. Lütfen dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")


def preprocess_data(data):
    """
    Veri ön işleme adımlarını içerir: eksik değerleri doldurma, kategorik değişkenleri sayısal hale getirme.
    Args:
        data (DataFrame): İşlenecek veri seti.
    Returns:
        DataFrame: İşlenmiş veri seti.
    """
    # 1. Eksik değerleri kontrol et ve doldur
    data = data.fillna(method='ffill')  # Eksik değerleri bir önceki değerle doldurma
    print("Eksik değerler dolduruldu.")

    # 2. Kategorik değişkenleri sayısal hale getirme
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
        print(f"{column} sütunu sayısal değerlere dönüştürüldü.")

    return data, label_encoders


def split_data(data, test_size=0.2, random_state=42):
    """
    Veriyi eğitim ve test setlerine böler.
    Args:
        data (DataFrame): İşlenmiş veri seti.
        test_size (float): Test setinin oranı.
        random_state (int): Rastgelelik kontrolü için sabit bir değer.
    Returns:
        Tuple: Eğitim ve test setleri (X_train, X_test, y_train, y_test).
    """
    X = data.drop(columns=['Churn Label'])  # Hedef sütunu dışındaki veriler
    y = data['Churn Label']  # Hedef sütun
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("Veri eğitim ve test setlerine ayrıldı.")
    return X_train, X_test, y_train, y_test


def main():
    # 1. Veri Yükleme
    filepath = '//Decision_Trees/data/telco.csv'  # Veri setinin yolunu belirtin
    data = load_data(filepath)

    # 2. Veri Ön İşleme
    if data is not None:
        data, label_encoders = preprocess_data(data)

        # 3. Veriyi Eğitim ve Test Setlerine Ayırma
        X_train, X_test, y_train, y_test = split_data(data)

        # 4. İşlemlerin Doğruluğunu Test Etme
        print("Eğitim veri seti boyutu:", X_train.shape)
        print("Test veri seti boyutu:", X_test.shape)
        print("Ön işleme ve veri ayırma işlemleri başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
