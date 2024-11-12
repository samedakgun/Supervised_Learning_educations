# preprocessing.py (Güncellenmiş Versiyon)
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_data(data, scaling_method="standard"):
    """
    Veriyi temizler ve ön işler.

    Args:
    - data (DataFrame): İşlenecek veri kümesi
    - scaling_method (str): Ölçekleme türü, 'standard' veya 'minmax'

    Returns:
    - processed_data (DataFrame): İşlenmiş veri kümesi
    - scaler (Scaler): Kullanılan ölçekleyici nesne
    """
    # Eksik değerleri ortalama ile doldur
    data = data.fillna(data.mean())

    # Hedef sütunu ayırma (eğer varsa)
    if "MEDV" in data.columns:
        features = data.drop(columns=["MEDV"])  # Hedef değişkeni çıkarıyoruz
        target = data["MEDV"]
    else:
        features = data
        target = None

    # Ölçekleme yöntemi seçimi
    if scaling_method == "standard":
        scaler = StandardScaler()
    elif scaling_method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Geçersiz ölçekleme yöntemi. 'standard' veya 'minmax' olmalıdır.")

    # Özellikleri ölçeklendirme
    scaled_features = scaler.fit_transform(features)
    processed_data = pd.DataFrame(scaled_features, columns=features.columns)

    # Eğer hedef sütunu varsa, geri ekleyin
    if target is not None:
        processed_data["MEDV"] = target

    return processed_data, scaler
