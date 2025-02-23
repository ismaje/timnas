import streamlit as st
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Memuat model SVM dan TF-IDF vectorizer yang telah disimpan
with open('svm_model (1).pkl', 'rb') as f:
    svm_classifier = pickle.load(f)

print("Kelas model:", svm_classifier.classes_)

with open('tfidf_vectorizer (1).pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Fungsi untuk melakukan prediksi terhadap input teks
def predict_sentiment(input_text):
    input_tfidf = tfidf_vectorizer.transform([input_text])  # Mengubah teks input menjadi fitur TF-IDF
    prediction = svm_classifier.predict(input_tfidf)  # Prediksi menggunakan model SVM
    return 'Positif' if prediction[0] == 1 else 'Negatif'  # Mengembalikan hasil prediksi

# Header dan penjelasan aplikasi
st.title("Analisis Sentimen ⚽︎")
st.write("Aplikasi analisis sentimen menggunakan model SVM untuk memprediksi sentimen dari teks yang Anda masukkan.")


# Input dari pengguna
user_input = st.text_input("📝 Masukkan kalimat untuk diprediksi:", "")

# Tombol untuk memulai prediksi
if st.button('🔍 Prediksi Sentimen'):
    if user_input:  # Pastikan ada input dari pengguna
        sentiment = predict_sentiment(user_input)  # Prediksi sentimen
        
        # Menentukan warna dan emoji berdasarkan sentimen
        if sentiment == "Positif":
            st.success(f"😊 Sentimen dari kalimat: *{sentiment}*")
        else:
            st.error(f"😠 Sentimen dari kalimat: *{sentiment}*")
    else:
        st.warning("⚠ Silakan masukkan kalimat terlebih dahulu.")

st.markdown("---")
st.write("Isma Magfirotul Yuna 21.12.1871")
