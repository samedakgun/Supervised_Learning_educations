

# Lojistik Regresyon ile Spam Tespiti

Bu proje, lojistik regresyon modeli kullanarak spam e-postaları tespit etmeyi amaçlamaktadır. Etiketlenmiş bir e-posta veri seti ile model, bir e-postanın spam olup olmadığını içeriğine göre sınıflandırmak üzere eğitilir. Proje; veri ön işleme, model eğitimi, değerlendirme ve tahmin adımlarını içermektedir.

## Proje Yapısı

- `data/`
  - `emails.csv`: Eğitim ve değerlendirme için kullanılan veri seti.
- `notebooks/`
  - `eda_notebook.ipynb`: Keşifsel veri analizi için Jupyter not defteri.
- `models/`
  - `model.pkl`: Eğitilmiş modelin, vektörizer ve etiket kodlayıcı ile birlikte kaydedildiği dosya.
- `src/`
  - `data_preprocessing.py`: Veri ön işleme scripti.
  - `train_model.py`: Model eğitimi scripti.
  - `evaluate_model.py`: Model değerlendirme scripti.
  - `predict.py`: Yeni e-postalar üzerinde tahmin yapma scripti.
- `README.md`: Proje açıklaması ve kullanım talimatları.
- `requirements.txt`: Gerekli Python kütüphanelerini içeren dosya.

## Kurulum

1. Bu projeyi bilgisayarınıza klonlayın.
2. Sanal bir ortam oluşturun ve aktif hale getirin:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Windows için `venv\Scripts\activate`
    ```
3. Gerekli bağımlılıkları yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

## Kullanım

### 1. Veri Ön İşleme

Veri ön işleme adımını gerçekleştirmek için:
```bash
python src/data_preprocessing.py
```

### 2. Modeli Eğitme

Veri seti üzerinde lojistik regresyon modelini eğitmek için:
```bash
python src/train_model.py
```
Eğitilen model, `models/` dizininde `model.pkl` olarak kaydedilecektir.

### 3. Modeli Değerlendirme

Modelin doğruluk, hassasiyet, duyarlılık ve F1 skoru gibi performans metriklerini hesaplamak için:
```bash
python src/evaluation.py
```

### 4. Tahmin Yapma

Yeni bir e-posta metninin spam olup olmadığını tahmin etmek için:
```bash
python src/predict.py
```
Tahmin yapılacak metni `predict.py` dosyasındaki `sample_text` değişkenine girerek değiştirebilirsiniz.

## Veri Seti

`emails.csv` dosyası, eğitim ve değerlendirme için etiketlenmiş e-posta verilerini içerir. Her satırda `text` (e-posta içeriği) ve `label` (spam veya değil etiketi) olmak üzere iki sütun bulunmalıdır.

## Gereksinimler

Gerekli Python kütüphanelerini yüklemek için `requirements.txt` dosyasını kullanabilirsiniz.
```
