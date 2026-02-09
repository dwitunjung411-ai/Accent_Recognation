import numpy as np
import tensorflow as tf

# ... (pastikan variabel support_set, support_labels, n_way, classes sudah ada di memori)

if st.button("Extract Feature and Detect"):
    if uploaded_file is not None:
        try:
            with st.spinner('Sedang memproses audio...'):
                # 1. Ekstrak Fitur Audio Upload
                # Pastikan fungsi ini mengembalikan array 1D (misal shape: (193,))
                query_features = extract_feature(uploaded_file) 
                
                # 2. Reshape agar jadi (1, n_features)
                # Model butuh batch dimension walaupun cuma 1 file
                query_set_input = np.expand_dims(query_features, axis=0)
                
                # 3. KONVERSI KE TENSOR (Penting untuk TensorFlow)
                # Kita ubah semua ke Tensor agar tipe datanya sinkron
                support_set_tensor = tf.convert_to_tensor(support_set, dtype=tf.float32)
                query_set_tensor = tf.convert_to_tensor(query_set_input, dtype=tf.float32)
                support_labels_tensor = tf.convert_to_tensor(support_labels, dtype=tf.int32)
                
                # Jika n_way bentuknya scalar (angka biasa), biarkan saja, atau cast ke int
                n_way_val = int(n_way) 

                # 4. PREDIKSI (Bungkus dalam List)
                # Berdasarkan struktur 'def call' kamu, dia mengecek if inputs_data is list
                inputs_paket = [support_set_tensor, query_set_tensor, support_labels_tensor, n_way_val]
                
                # Panggil model
                # Kita pakai model.predict() atau model() langsung
                logits = model(inputs_paket)
                
                # 5. Ambil Hasil
                # Logits output shape: (1, n_way) -> Ambil index max
                pred_index = tf.argmax(logits, axis=1)[0].numpy()
                
                # Cek confidence score
                probs = tf.nn.softmax(logits)[0]
                confidence = probs[pred_index] * 100
                
                hasil_aksen = classes[pred_index]

            # Tampilkan Output
            st.success(f"🗣️ Aksen Terdeteksi: **{hasil_aksen}**")
            st.info(f"📊 Tingkat Keyakinan: {confidence:.2f}%")
            
            # (Opsional) Tampilkan probabilitas semua kelas
            with st.expander("Lihat Detail Probabilitas"):
                for i, class_name in enumerate(classes):
                    st.write(f"{class_name}: {probs[i]*100:.2f}%")

        except Exception as e:
            # FITUR DEBUGGING: Tampilkan pesan error lengkap jika gagal
            st.error("Terjadi Error Sistem:")
            st.write(e)
            
            st.write("--- Data Debug ---")
            st.write(f"Shape Support Set: {np.shape(support_set)}")
            st.write(f"Shape Query Set: {np.shape(query_set_input)}")
            st.write("Coba cek apakah jumlah fitur (kolom terakhir) sama?")
    else:
        st.warning("Mohon upload file audio (.wav/.mp3) dulu.")
