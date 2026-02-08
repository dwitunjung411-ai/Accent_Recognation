@st.cache_resource
def load_centroids():
    return np.load("accent_centroids.npy", allow_pickle=True).item()
def predict_accent(audio_path, model):
    try:
        mfcc = extract_mfcc(audio_path)
        mfcc = np.expand_dims(mfcc, axis=0)

        # Ambil embedding
        embedding = model.predict(mfcc)
        embedding = np.asarray(embedding).squeeze()

        centroids = load_centroids()

        distances = {}
        for cls, centroid in centroids.items():
            centroid = np.asarray(centroid).squeeze()

            if embedding.shape != centroid.shape:
                raise ValueError(
                    f"Shape mismatch: embedding {embedding.shape} vs centroid {centroid.shape} for class {cls}"
                )

            distances[cls] = np.linalg.norm(embedding - centroid)

        predicted_class = min(distances, key=distances.get)
        return predicted_class

    except Exception as e:
        return f"Error Analisis: {e}"
