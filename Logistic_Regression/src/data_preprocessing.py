import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re


def load_and_preprocess_data(file_path):
    # Veriyi yükleme
    data = pd.read_csv(file_path)

    # 'text' ve 'label' sütunlarının olup olmadığını kontrol etme
    if 'text' not in data.columns or 'spam' not in data.columns:
        raise ValueError("Veri kümesi 'text' ve 'label' sütunlarını içermelidir.")

    # Boş değerlere sahip satırları kaldırma
    data.dropna(subset=['text', 'spam'], inplace=True)

    # Metin temizleme fonksiyonu
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)  # Çoklu boşlukları tek boşluğa indirgeme
        text = re.sub(r'[^A-Za-z0-9ğüşöçİĞÜŞÖÇ ]', '', text)  # Özel karakterleri kaldırma
        text = text.strip()  # Baş ve sondaki boşlukları silme
        return text.lower()  # Küçük harfe çevirme

    # 'text' sütunundaki her bir metin verisine temizleme işlemini uygulama
    data['text'] = data['text'].apply(clean_text)

    # Etiketleri sayısal hale getirme
    label_encoder = LabelEncoder()
    data['spam'] = label_encoder.fit_transform(data['spam'])

    # Metin verilerini vektörize etme
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['text'])
    y = data['spam']

    return X, y, vectorizer, label_encoder


if __name__ == "__main__":
    # Örnek kullanım
    X, y, vectorizer, label_encoder = load_and_preprocess_data('//Logistic_Regression/data/emails.csv')
    if X is not None:
        print("Ön işleme işlemleri tamamlandı.")
