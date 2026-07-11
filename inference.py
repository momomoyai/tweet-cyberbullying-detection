import pickle
import numpy as np
import re
import nltk
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

with open('lgbm_binary.pkl', 'rb') as f:
    model_binary = pickle.load(f)

with open('lgbm_tipe.pkl', 'rb') as f:
    model_multiclass = pickle.load(f)

with open('tfidf_dict.pkl', 'rb') as f:
    tfidf_dict = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

w2v_model = Word2Vec.load('w2v_model.gensim')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize(text):
    """
    Melakukan pembersihan teks yang identik dengan proses training teks di notebook.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # Menyisakan huruf a-z
    
    words = word_tokenize(text)
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return " ".join(cleaned_words)

def get_weighted_vectors(corpus, w2v_model, tfidf_dict):
    """
    Membentuk matriks embedding berbasis rata-rata bobot TF-IDF dari teks input.
    """
    vectors = []
    v_size = w2v_model.vector_size
    
    for sentence in corpus:
        words = sentence.split()
        weighted_vector = np.zeros(v_size)
        total_weight = 0.0
        
        for word in words:
            if word in w2v_model.wv and word in tfidf_dict:
                weighted_vector += w2v_model.wv[word] * tfidf_dict[word]
                total_weight += tfidf_dict[word]
        
        if total_weight > 0:
            weighted_vector /= total_weight
            
        vectors.append(weighted_vector)
        
    return np.array(vectors)

def predict_pipeline(user_input):
    """
    Mengalirkan input user ke Model 1 (Biner).
    Jika True, dilanjutkan ke Model 2 (Multi-kelas) untuk mendeteksi tipe spesifik.
    """
    # 1. Preprocessing teks mentah
    cleaned_text = clean_and_lemmatize(user_input)
    
    # Validasi jika input kosong atau hanya berisi karakter ilegal/stopword
    if not cleaned_text.strip():
        return {
            "status": "error",
            "message": "Input tidak valid setelah prapemrosesan teks."
        }
    
    # 2. Konversi teks ke bentuk vektor numerik
    X_vector = get_weighted_vectors([cleaned_text], w2v_model, tfidf_dict)

    feature_names = model_binary.feature_name_ # Jika atribut ini ada
    X_vector_df = pd.DataFrame(X_vector, columns=feature_names)
    
    # 3. Prediksi Tahap 1: Apakah Cyberbullying?
    is_cyberbullying_pred = model_binary.predict(X_vector_df)[0]
    
    # Konversi hasil prediksi biner ke boolean sejati
    is_cyberbullying = bool(is_cyberbullying_pred)
    
    response = {
        "status": "success",
        "processed_text": cleaned_text,
        "is_cyberbullying": is_cyberbullying,
        "cyberbullying_type": None  # Default jika bukan merupakan cyberbullying
    }
    
    # 4. Prediksi Tahap 2: Jika True, jalankan klasifikasi multi-kelas
    if is_cyberbullying:
        multiclass_pred = model_multiclass.predict(X_vector_df)[0]
        
        # Kembalikan representasi angka prediktif menjadi label teks teks asli
        type_label = label_encoder.inverse_transform([multiclass_pred])[0]
        response["cyberbullying_type"] = type_label
        
    return response