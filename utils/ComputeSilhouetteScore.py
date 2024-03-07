from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def compute_silhouette_score(params, sample_data):
    k, init_method = params
    kmeans = KMeans(n_clusters=k, init=init_method, random_state=42, n_init=10)
    labels = kmeans.fit_predict(sample_data)
    silhouette_avg = silhouette_score(sample_data, labels)
    return silhouette_avg, k, init_method