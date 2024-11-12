# predict.py
import joblib
import pandas as pd
from preprocessing import preprocess_data

def predict_new_data(new_data, model_path='/Users/samedakgun/PycharmProjects/Supervised_Learning/Linear_Regression/models/trained_model.pkl'):
    """
    Eğitilmiş model ile yeni veri üzerinde tahmin yapar.

    Args:
    - new_data (DataFrame): Tahmin yapılacak yeni veri
    - model_path (str): Kaydedilen modelin yolu

    Returns:
    - Series: Tahmin edilen değerler
    """
    # Modeli yükle
    model = joblib.load(model_path)

    # Yeni veri üzerinde tahmin yap
    predictions = model.predict(new_data)
    return predictions

# Kullanım
if __name__ == "__main__":
    # Örnek yeni veri
    sample_data = pd.DataFrame({
        'CRIM': [0.00632],
        'ZN': [18.0],
        'INDUS': [2.31],
        'CHAS': [0],
        'NOX': [0.538],
        'RM': [6.575],
        'AGE': [65.2],
        'DIS': [4.09],
        'RAD': [1],
        'TAX': [296],
        'PTRATIO': [15.3],
        'B': [396.9],
        'LSTAT': [4.98]
    })

    # preprocess_data ile yeni veriyi ölçeklendirme
    processed_sample_data, _ = preprocess_data(sample_data, scaling_method="standard")

    # Tahmin yap ve sonuçları yazdır
    predictions = predict_new_data(processed_sample_data)
    print("Tahmin edilen ev fiyatı:", predictions)
