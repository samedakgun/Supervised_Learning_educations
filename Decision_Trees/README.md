# Customer Churn Prediction with Decision Trees

Bu proje, müşteri terk (churn) analizini yapmak için bir Karar Ağacı modeli oluşturur ve test eder. Proje, bir veri setini kullanarak model eğitimi, değerlendirme ve tahmin işlemlerini gerçekleştirir.

## Dosya Yapısı
- `data/`: Veri dosyaları
- `models/`: Kaydedilen model dosyaları
- `src/`: Kod dosyaları
  - `data_preprocessing.py`: Veri yükleme ve işleme adımları
  - `model_training.py`: Model eğitimi ve kaydetme işlemleri
  - `evaluation.py`: Model değerlendirme metriklerini hesaplar ve gösterir
  - `main.py`: Kendi değerlerinle tahmin yapmak için interaktif bir script

## Gereksinimler
Proje, Python ve çeşitli kütüphaneleri gerektirir. Gereksinimleri kurmak için aşağıdaki komutu çalıştırın:
```bash
pip install -r requirements.txt
