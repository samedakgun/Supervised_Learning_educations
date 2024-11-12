import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Veri başarıyla yüklendi. İlk birkaç satır:")
    print(data.head())
    return data


def preprocess_data(data):
    # Gereksiz sütunları kaldırma (örneğin 'id' ve 'Unnamed: 32')
    if 'id' in data.columns:
        data = data.drop(['id'], axis=1)
        print("ID sütunu kaldırıldı.")

    if 'Unnamed: 32' in data.columns:
        data = data.drop(['Unnamed: 32'], axis=1)
        print("Unnamed: 32 sütunu kaldırıldı.")

    print("Eksik değer sayısı (öncesi):")
    print(data.isna().sum().sum())

    # Sadece sayısal sütunlardaki eksik değerleri ortalama ile doldurma
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    print("Eksik değer sayısı (doldurma sonrası):")
    print(data.isna().sum().sum())

    # Hedef değişkeni binary olarak kodlama
    if 'diagnosis' in data.columns:
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        print("Diagnosis sütunu binary olarak kodlandı.")

    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    print("Ön işleme tamamlandı. X ve y döndürülüyor.")
    return X, y


def split_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("Veri eğitim ve test setlerine ayrıldı.")
    print(f"Eğitim seti boyutu: {X_train.shape}, Test seti boyutu: {X_test.shape}")
    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    data_path = '//SVM/data/data.csv'
    data = load_data(data_path)
    X, y = preprocess_data(data)
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = split_data(X, y)
