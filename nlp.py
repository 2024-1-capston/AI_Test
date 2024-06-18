from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)

model = SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens')

def cluster_sentences(sentences, num_clusters=2):
    embeddings = model.encode(sentences, show_progress_bar=False)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    return cluster_centers, cluster_labels

def summarize_sentences(sentences, num_clusters=2, num_sentences_per_cluster=1):
    cluster_centers, cluster_labels = cluster_sentences(sentences, num_clusters)
    summarized_sentences = []
    
    for cluster_idx in range(num_clusters):
        cluster_sentences_in_cluster = [sentences[i] for i, label in enumerate(cluster_labels) if label == cluster_idx]
        if cluster_sentences_in_cluster:
            summarized_sentences.extend(cluster_sentences_in_cluster[:num_sentences_per_cluster])
    
    return summarized_sentences

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()

    text = data['text']

    try:
        sentences = text.split('.') 
        summarized_sentences = summarize_sentences(sentences, num_clusters=2, num_sentences_per_cluster=1)

        summary = ' '.join(summarized_sentences)

        return jsonify({'summary': summary})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
