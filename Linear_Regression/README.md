## Proje Dosya Yapısı

```plaintext
house_price_prediction
├── data            # Veri kümesi dosyaları
├── models          # Eğitilmiş model dosyaları
├── src             # Kod dosyaları (veri işleme, model eğitimi, değerlendirme)
├── README.md       # Proje hakkında bilgi dosyası
└── config          # Yapılandırma dosyaları (config.py, config.json)
```

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
pandas
numpy
joblib
scikit-learn

## Kullanım

1. **Veri Yükleme ve Ön İşleme**: Veri kümesini `data_loader.py` ve `preprocessing.py` dosyaları ile yükleyin ve ön işleme yapın.
2. **Model Eğitimi**: `train.py` dosyasını çalıştırarak modelinizi eğitin ve kaydedin.
   ```bash
   python src/train_model.py
   ```
3. **Model Değerlendirme**: Eğitilen modelin performansını `evaluate.py` dosyası ile test edin.
   ```bash
   python src/evaluation.py
   ```
4. **Tahmin Yapma**: Yeni bir veri üzerinden tahmin yapmak için `predict.py` dosyasını kullanın.
   ```bash
   python src/predict.py
   ```

## Yapılandırma Dosyaları

- **config.json**: Proje ayarları ve parametrelerin saklandığı dosya.
- **config.py**: Python formatında yapılandırma dosyası, sabit ayarları içerir.

## Kullanılan Kütüphaneler

- `pandas`: Veri işleme ve analiz.
- `scikit-learn`: Model eğitimi, değerlendirme ve ölçeklendirme.
- `joblib`: Modeli kaydetmek ve yüklemek için.


