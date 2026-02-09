import numpy as np
import tensorflow as tf

# Tombol Prediksi
if st.button("Extract Feature and Detect"):
    if uploaded_file is not None:
        try:
            # 1. Tampilkan status loading
            with st.spinner('Sedang mengekstrak fitur audio...'):
                # Pastikan fungsi extract_feature ada dan mengembalikan array 1D
                # Sesuaikan nama fungsi ini dengan yang ada di kodemu (misal: extract_feature_v2)
                query_features = extract_feature(uploaded_file) 
            
            st.success("Ekstraksi fitur berhasil!")
            
            # 2. Persiapan Data untuk Model FSL
            # Model FSL butuh dimensi (batch_size, n_features)
            # Kita tambah dimensi agar jadi (1, 193) misalnya
            query_set_input = np.expand_dims(query_features, axis=0)
            
            # DEBUG: Tampilkan shape data untuk memastikan benar
            st.write("--- Debug Info ---")
            st.write(f"Shape Support Set: {support_set.shape}") 
            st.write(f"Shape Query Set (Input): {query_set_input.shape}")
            
            # 3. Prediksi
            # Kita panggil model.call secara manual jika model.predict bermasalah
            # Ini memotong jalur standar Keras dan langsung ke logika modelmu
            
            # Pastikan variable ini sudah ada sebelumnya:
            # n_way = 5  (misalnya)
            # support_labels = ... (label angka 0-4)
            
            logits = model.call(
                support_set,        # Support set (referensi)
                query_set_input,    # Query set (file upload)
                support_labels,     # Label support set
                n_way               # Jumlah kelas
            )
            
            # 4. Ambil Hasil
            # Logits biasanya bentuknya (1, n_way), kita ambil index terbesarnya
            predicted_index = int(tf.argmax(logits, axis=1)[0])
            
            # Mapping ke nama kelas (Pastikan variable 'classes' ada)
            # Contoh: classes = ['Batak', 'Jawa', 'Sunda', 'Minang', 'Madura']
            hasil_prediksi = classes[predicted_index]
            confidence = tf.nn.softmax(logits)[0][predicted_index] * 100
            
            # Tampilkan Hasil
            st.success(f"✅ Aksen Terdeteksi: **{hasil_prediksi}**")
            st.info(f"Confidence: {confidence:.2f}%")
            
        except Exception as e:
            st.error("Terjadi Kesalahan (Error):")
            st.code(e) # Ini akan menampilkan pesan error lengkap
            st.warning("""
            Tips Perbaikan:
            1. Cek 'Debug Info' di atas. Apakah shape Query Set sama jumlah fiturnya dengan Support Set?
            2. Pastikan 'support_set' dan 'support_labels' sudah diload dengan benar di awal script.
            """)
    else:
        st.warning("Silakan upload file audio terlebih dahulu.")
