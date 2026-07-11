import streamlit as st
from inference import predict_pipeline

st.set_page_config(
    page_title="Tweet Cyberbullying Detection", 
    page_icon="🕊️",
    layout="centered"
)

st.title("🕊️ Deteksi Cyberbullying Tweet")
st.write("Sistem ini menggunakan arsitektur model cascading LightGBM untuk mendeteksi cyberbullying dan mengklasifikasikan tipe perundungannya berdasarkan tweet.")
st.write(" ")
st.write("Delivered by: Group 3 - Aletheia")

user_input = st.text_area(
    "Masukkan teks untuk dianalisis:", 
    height=150, 
    placeholder="Ketik komentar atau teks di sini..."
)

if st.button("Analisis Teks", type="primary"):
    if not user_input.strip():
        st.warning("Silakan masukkan teks terlebih dahulu sebelum melakukan analisis!")
    else:
        with st.spinner("Memproses teks dan menjalankan model inferensi..."):
            
            result = predict_pipeline(user_input)
            
            if result["status"] == "error":
                st.error(f"Terjadi kesalahan: {result['message']}")
            else:
                st.subheader("Hasil Analisis:")
                st.write("Note: Hasil prediksi model tidak selalu akurat. Tanggung jawab penggunaan informasi dari model ditanggung sepenuhnya oleh pengguna.")
                
                if result["is_cyberbullying"]:
                    st.error("⚠️ **Terdeteksi Indikasi Cyberbullying!**")
                    
                    tipe_cb = result['cyberbullying_type']
                    st.info(f"**Klasifikasi Tipe:** {tipe_cb.upper()}")
                else:
                    st.success("✅ **Teks aman.** Tidak terdeteksi adanya indikasi cyberbullying.")
                
                with st.expander("Lihat detail prapemrosesan teks (Debug)"):
                    st.text("Teks setelah dibersihkan dan dilemmatisasi:")
                    st.code(result["processed_text"])