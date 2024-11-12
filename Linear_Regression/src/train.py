# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def train_model(data, model_save_path='/Users/samedakgun/PycharmProjects/Supervised_Learning/Linear_Regression/models/trained_model.pkl'):
    """
    Veriyi eğitim ve test setlerine ayırarak modeli eğitir ve kaydeder.

    Args:
    - data (DataFrame): İşlenmiş veri kümesi
    - model_save_path (str): Eğitilmiş modelin kaydedileceği dosya yolu

    Returns:
    - model: Eğitilmiş model
    """
    # Özellik ve hedef değişkeni ayırma
    X = data.drop(columns=["MEDV"])
    y = data["MEDV"]

    # Eğitim ve test setlerine ayırma (%80 eğitim, %20 test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model oluşturma ve eğitme
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Eğitilmiş modeli kaydetme
    joblib.dump(model, model_save_path)
    print(f"Model başarıyla '{model_save_path}' dosyasına kaydedildi.")

    return model, X_test, y_test

# Kullanım
if __name__ == "__main__":
    data = pd.read_csv('//Linear_Regression/data/boston.csv')
    # preprocessing.py modülündeki preprocess_data fonksiyonunu çağırarak veri işleme yapılabilir
    from preprocessing import preprocess_data
    processed_data, _ = preprocess_data(data, scaling_method="standard")
    train_model(processed_data)
