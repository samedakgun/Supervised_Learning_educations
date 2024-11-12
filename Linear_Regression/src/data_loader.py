# data_loader.py
import pandas as pd

def load_data(file_path):
    """
    Veri kümesini yükler ve temel analiz yapar.

    Args:
    - file_path (str): Veri kümesi dosyasının yolu

    Returns:
    - data (DataFrame): Yüklenen veri kümesi
    """
    # Veri kümesini yükle
    data = pd.read_csv(file_path)

    # Veri hakkında temel bilgileri görüntüle
    print("Veri hakkında genel bilgi:")
    print(data.info())
    print("\nİlk birkaç satır:")
    print(data.head())

    # Temel istatistikleri görüntüle
    print("\nTemel istatistikler:")
    print(data.describe())

    # Eksik değerleri kontrol et
    missing_values = data.isnull().sum()
    print("\nEksik değer sayısı:")
    print(missing_values[missing_values > 0])

    return data

# Kullanım
if __name__ == "__main__":
    file_path = '/Linear_Regression/data/boston.csv'  # Veri kümesi dosyasının yolu
    data = load_data(file_path)
