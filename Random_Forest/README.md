# Bank Marketing Random Forest Classifier

Bu proje, Bank Marketing veri seti üzerinde Random Forest algoritması kullanarak müşteri hedefleme tahminleri yapan bir sınıflandırma modeli oluşturur. Model, bankacılık kampanyalarına müşterinin olumlu yanıt verip vermeyeceğini tahmin eder.


## Kurulum

1. Bu projeyi yerel makinenize klonlayın veya indirin.
2. Gerekli Python kütüphanelerini kurmak için terminalde aşağıdaki komutu çalıştırın:
   
   ```bash
   pip install -r requirements.txt

Kullanım
1. Veriyi İşleme
Veriyi hazırlamak için data_preprocessing.py dosyasını çalıştırın. Bu işlem, veriyi temizleyecek, eksik değerleri dolduracak, kategorik değişkenleri sayısal değerlere dönüştürecek ve label_encoders.pkl dosyasına kaydedecektir.

bash
Kodu kopyala
python src/data_preprocessing.py
2. Model Eğitimi
Random Forest modelini eğitmek ve kaydetmek için model_training.py dosyasını çalıştırın. Eğitilen model models/random_forest_model.pkl olarak kaydedilecektir.

bash
Kodu kopyala
python src/model_training.py
3. Model Değerlendirme
Eğitilen modelin performansını değerlendirmek için evaluation.py dosyasını çalıştırabilirsiniz.

bash
Kodu kopyala
python src/evaluation.py
4. Tahmin Yapma
Bir müşteri hakkında tahminde bulunmak için main.py dosyasını çalıştırın. Program, bazı özellikleri otomatik olarak kullanacak ve tanımlı olmayan kategorik değerler için 'unknown' veya en sık görülen değeri kullanacaktır.

bash
Kodu kopyala
python src/main.py
Gereksinimler
Python 3.6+
Gerekli kütüphaneler requirements.txt dosyasında listelenmiştir.
Veri Seti
Bu proje, UCI Machine Learning Repository'den alınan Bank Marketing veri setini kullanır. Veri setinde, bankacılık kampanyalarına müşteri yanıtı gibi bilgileri içeren çeşitli özellikler bulunmaktadır.

