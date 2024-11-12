import joblib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """
    Modelin performansını test verileri üzerinde değerlendirir.

    Args:
    - model: Eğitilmiş model
    - X_test (DataFrame): Test özellikleri
    - y_test (Series): Test hedef değişkeni

    Returns:
    - dict: Modelin performans metriklerini içeren sözlük
    """
    # Tahmin yapma
    y_pred = model.predict(X_test)

    # Performans metrikleri
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Sonuçları ekrana yazdırma
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R²): {r2}")

    return {"MSE": mse, "RMSE": rmse, "R2": r2}

# Kullanım
if __name__ == "__main__":
    # Eğitilmiş modeli yükleyin
    model_path = '//Linear_Regression/models/trained_model.pkl'
    model = joblib.load(model_path)

    # train_model.py içindeki train_model fonksiyonundan test verilerini alın
    from train import train_model
    _, X_test, y_test = train_model(pd.read_csv('//Linear_Regression/data/boston.csv'))

    evaluate_model(model, X_test, y_test)
