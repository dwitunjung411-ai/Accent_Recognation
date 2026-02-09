import numpy as np
import tensorflow as tf

# --- MULAI COPY DARI SINI ---
if uploaded_file is not None:
    # 1. Ekstrak fitur dari file yang diupload
    # (Pastikan fungsi extract_feature_v2 atau extract_feature sudah didefinisikan)
    x_query = extract_feature(uploaded_file) 
    
    # 2. Reshape Query agar memiliki dimensi batch (Contoh: dari (193,) menjadi (1, 193))
    # Ini WAJIB agar dianggap sebagai 1 sample data
    query_set_input = np.expand_dims(x_query, axis=0)
    
    # 3. Siapkan semua input yang dibutuhkan model 'call'
    # Model FSL kamu butuh 4 input sekaligus dalam satu list
    input_paket = [
        support_set_features,  # Data referensi (Support Set)
        query_set_input,       # Data upload (Query Set) <-- INI YANG TADI MISSING
        support_labels,        # Label angka untuk support set
        n_way                  # Jumlah kelas (misal: 5)
    ]

    # 4. Lakukan Prediksi
    # Kita panggil model(input_paket) langsung agar masuk ke logika 'call' yang melakukan unpacking
    try:
        logits = model(input_paket)
        
        # 5. Ambil hasil prediksi (Index kelas dengan probabilitas tertinggi)
        prediction_index = np.argmax(logits.numpy(), axis=1)[0]
        
        # Mapping ke nama kelas (Pastikan variable 'classes' berisi nama aksen, misal ['Jawa', 'Sunda', ...])
        hasil_prediksi = classes[prediction_index]
        
        # Tampilkan Hasil di Streamlit
        st.success(f"Aksen Terdeteksi: {hasil_prediksi}")
        
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
        st.write("Debug shape:", query_set_input.shape)

# --- SELESAI COPY ---
