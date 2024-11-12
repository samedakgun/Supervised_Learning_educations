# data_preprocessing.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    """
    Veriyi dosyadan yükler.
    Args:
        filepath (str): Veri dosyasının yolu.
    Returns:
        DataFrame: Veri çerçevesi.
    """
    data = pd.read_csv(filepath)
    print("Veri başarıyla yüklendi.")
    return data


def preprocess_data(data):
    """
    Veriyi temizler, eksik değerleri ortalama ile doldurur ve kategorik
    özellikleri sayısal değerlere dönüştürür.
    Args:
        data (DataFrame): Orijinal veri çerçevesi.
    Returns:
        X (DataFrame): Özellikler.
        y (Series): Hedef değişken.
    """
    # Hedef değişkeni ayırma
    y = data['deposit'].apply(lambda x: 1 if x == 'yes' else 0)
    X = data.drop(columns=['deposit'])

    # Eksik değerleri kontrol etme ve ortalama ile doldurma
    if X.isnull().sum().sum() > 0:
        print("Eksik değerler bulundu, ortalama ile dolduruluyor.")
        for column in X.select_dtypes(include=['float64', 'int64']).columns:
            mean_value = X[column].mean()
            X[column].fillna(mean_value, inplace=True)
            print(f"{column} sütunundaki eksik değerler {mean_value:.2f} ortalaması ile dolduruldu.")
    else:
        print("Eksik değer bulunamadı.")

    # Kategorik verileri dönüştürme
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # Label encoderları kaydet
    joblib.dump(label_encoders, '../models/label_encoders.pkl')
    print("Label encoders 'label_encoders.pkl' dosyasına kaydedildi.")

    return X, y, label_encoders


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Veriyi eğitim ve test setlerine ayırır.
    Args:
        X (DataFrame): Özellikler.
        y (Series): Hedef değişken.
        test_size (float): Test setinin boyutu.
        random_state (int): Rastgelelik için sabit değer.
    Returns:
        X_train, X_test, y_train, y_test: Eğitim ve test setleri.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("Veri eğitim ve test setlerine ayrıldı.")
    return X_train, X_test, y_train, y_test


def main():
    # Veri yolunu ayarla
    filepath = '//Random_Forest/data/bank.csv'

    # Veriyi yükle
    data = load_data(filepath)

    # Veriyi temizle ve ön işlemleri yap
    X, y, label_encoders = preprocess_data(data)

    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Eğitim ve test setlerini kaydet veya döndür
    # (Bu örnekte kaydetme işlemi yok; model eğitimi sırasında kullanılacak)
    return X_train, X_test, y_train, y_test, label_encoders


if __name__ == "__main__":
    main()
