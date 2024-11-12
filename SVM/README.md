
### `README.md`

Bu dosya, projenizin amacı, kurulum adımları, kullanım talimatları ve projenin nasıl çalıştırılacağını açıklayan temel bilgileri içermelidir.

```markdown
# SVM Cancer Classification Project

Bu proje, meme kanseri teşhisi için Destek Vektör Makineleri (SVM) algoritmasını kullanarak sınıflandırma yapmaktadır. Proje, verilen özellikler üzerinden kanserli (Malignant) veya kanserli olmayan (Benign) durumlarını tahmin eder.

## Proje Yapısı

```
SVM_Cancer_Classification/
├── data/
│   └── data.csv                 # Veri seti
├── models/
│   └── svm_model.pkl            # Eğitilmiş model dosyası
├── src/
│   ├── data_preprocessing.py    # Veri temizleme ve ön işleme
│   ├── model_training.py        # Model eğitimi
│   ├── evaluation.py            # Model değerlendirme
├── main.py                      # Ana dosya
├── README.md                    # Proje açıklaması
└── requirements.txt             # Gereksinimler
```

## Kurulum

Bu projeyi çalıştırmak için öncelikle bağımlılıkları yüklemeniz gerekmektedir. `requirements.txt` dosyasını kullanarak gerekli paketleri yükleyebilirsiniz:

```bash
pip install -r requirements.txt
```

## Kullanım

1. `data/data.csv` konumunda bir veri seti bulundurun. Örnek veri seti meme kanseri teşhisi için gerekli özellikleri içerir.
2. Aşağıdaki komutla `main.py` dosyasını çalıştırarak model eğitimi ve tahmin işlemini gerçekleştirebilirsiniz:

```bash
python main.py
```

3. Model eğitildikten sonra, `models/svm_model.pkl` olarak kaydedilecektir.
4. `main.py` içinde belirtilen örnek veri ile tahmin işlemi yapılacaktır.

## Dosya Açıklamaları

- **data_preprocessing.py**: Veri yükleme, temizleme ve eğitim/test setine ayırma işlemleri bu dosyada yapılır.
- **model_training.py**: SVM modelini eğitmek ve eğitilen modeli kaydetmek için gerekli fonksiyonları içerir.
- **evaluation.py**: Modelin performansını test verisi üzerinde değerlendirir.
- **main.py**: Tüm adımları birleştirerek modeli eğitir, değerlendirir ve yeni örnekle test eder.
- **requirements.txt**: Proje için gerekli Python paketlerinin listesi.
