import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score,davies_bouldin_score
import re
from collections import Counter
import warnings
from nltk.corpus import stopwords
import nltk
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath='dataset.csv'):
    df = pd.read_csv(filepath)
    def preprocess_text(text):
        if pd.isna(text):
            return []
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Split into words and remove empty strings
        nltk.download('stopwords')
        STOPWORDS = set(stopwords.words("english"))
        words = [w for w in text.split() if len(w) > 2 and w not in STOPWORDS]
        return words
    
    df['processed_words'] = df['message'].apply(preprocess_text)
    
    return df

def train_word2vec(all_words, vector_size=100, min_count=2, window=5):
    model = Word2Vec(
        sentences=all_words,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        workers=4,
        seed=42
    )
    return model

def cluster_words(word2vec_model, n_clusters):
    # Get all words and their vectors
    words = word2vec_model.wv.index_to_key
    word_vectors = np.array([word2vec_model.wv[word] for word in words])
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    word_clusters = kmeans.fit_predict(word_vectors)
    
    # Create word to cluster mapping
    word_to_cluster = {word: cluster for word, cluster in zip(words, word_clusters)}
    
    # Calculate silhouette score for clustering quality
    sil_score = silhouette_score(word_vectors, word_clusters)
    #dav_score = davies_bouldin_score(word_vectors, word_clusters)
    
    return word_to_cluster, sil_score

def create_bow_vectors(df, word_to_cluster, n_clusters):
    bow_vectors = []
    
    for words in df['processed_words']:
        # Initialize vector with zeros
        vector = np.zeros(n_clusters)
        
        # Count words in each cluster bin
        for word in words:
            if word in word_to_cluster:
                cluster = word_to_cluster[word]
                vector[cluster] += 1
        
        # Normalize by total number of words
        if sum(vector) > 0:
            vector = vector / sum(vector)
        
        bow_vectors.append(vector)
    
    return np.array(bow_vectors)

def cluster_documents(bow_vectors, n_doc_clusters=5):
    kmeans = KMeans(n_clusters=n_doc_clusters, random_state=42, n_init=10)
    doc_clusters = kmeans.fit_predict(bow_vectors)
    
    # Calculate silhouette score
    sil_score = silhouette_score(bow_vectors, doc_clusters)
    dav_score = davies_bouldin_score(bow_vectors, doc_clusters)
    return doc_clusters, sil_score,dav_score

def analyze_clusters(df, doc_clusters, word_to_cluster):
    df['doc_cluster'] = doc_clusters
    cluster_stats = []
    for cluster_id in range(max(doc_clusters) + 1):
        cluster_docs = df[df['doc_cluster'] == cluster_id]
        
        # Get most common words in cluster
        all_words = []
        for words in cluster_docs['processed_words']:
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(10)
        
        stats = {
            'cluster_id': cluster_id,
            'num_documents': len(cluster_docs),
            'top_words': top_words,
            'sample_messages': cluster_docs['message'].head(3).tolist()
        }
        cluster_stats.append(stats)
    
    return cluster_stats

def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('dataset.csv')
    all_words = df['processed_words'].tolist()
    
    # Three different configurations matching Doc2Vec dimensions
    # The final document vectors will have these dimensions
    doc_vector_dimensions = [50, 100, 200]  # Same as Doc2Vec
    results = []
    
    for target_dim in doc_vector_dimensions:
        print(f"\n{'='*60}")
        print(f"Creating {target_dim}-dimensional document vectors")
        print('='*60)
        
        # Train Word2Vec model 
        word_vector_size = target_dim
        print(f"Training Word2Vec model with {word_vector_size} dimensions...")
        word2vec_model = train_word2vec(all_words, vector_size=word_vector_size)
        
        # Number of word clusters = target document vector dimension
        n_clusters = target_dim  # This ensures document vectors have target_dim dimensions
        
        # Cluster words
        print(f"Clustering words into {n_clusters} bins...")
        print(f"This will create {n_clusters}-dimensional document vectors")
        word_to_cluster, word_cluster_score = cluster_words(word2vec_model, n_clusters)
        
        # Create BoW vectors (these will have n_clusters dimensions)
        print(f"Creating {n_clusters}-dimensional Bag-of-Words vectors...")
        bow_vectors = create_bow_vectors(df, word_to_cluster, n_clusters)
        print(f"Document vector shape: {bow_vectors.shape}")
        
        print("Clustering documents...")
        doc_clusters, doc_cluster_score,doc_dav_score = cluster_documents(bow_vectors, n_doc_clusters=5)
        
        cluster_stats = analyze_clusters(df, doc_clusters, word_to_cluster)
        
        result = {
            'doc_vector_dimension': target_dim,
            'word_vector_size': word_vector_size,
            'n_word_clusters': n_clusters,
            'word_clustering_silhouette': word_cluster_score,
            'doc_clustering_silhouette': doc_cluster_score,
            "doc_clustering_davies":doc_dav_score,
            'cluster_stats': cluster_stats
        }
        results.append(result)
        
        print(f"\nResults for {target_dim}-dimensional document vectors:")
        print(f"  Word vector size used: {word_vector_size}")
        print(f"  Number of word clusters (K): {n_clusters}")
        print(f"  Word clustering silhouette score: {word_cluster_score:.4f}")
        print(f"  Document clustering silhouette score: {doc_cluster_score:.4f}")
        print(f"\nDocument clusters analysis:")
        
        for stat in cluster_stats:
            print(f"\n  Cluster {stat['cluster_id']}:")
            print(f"    Number of documents: {stat['num_documents']}")
            print(f"    Top words: {[word for word, _ in stat['top_words'][:5]]}")
            print(f"    Sample message: {stat['sample_messages'][0][:100]}...")
    
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    
    # Identify best configuration
    best_config = max(results, key=lambda x: x['doc_clustering_silhouette'])
    
    # Prepare summary text
    summary_lines = []
    summary_lines.append("\nBest configuration based on document clustering quality:")
    summary_lines.append(f"  Document vector dimension: {best_config['doc_vector_dimension']}")
    summary_lines.append(f"  Document clustering silhouette score: {best_config['doc_clustering_silhouette']:.4f}")
    summary_lines.append(f"  Document clustering silhouette score: {best_config['doc_clustering_davies']:.4f}")
    
    summary_lines.append("\nSummary of all configurations:")
    summary_lines.append("(These dimensions match the Doc2Vec experiments)")
    
    for result in results:
        summary_lines.append(f"\nDocument vector dimension: {result['doc_vector_dimension']}")
        summary_lines.append(f"  - Number of word clusters (K): {result['n_word_clusters']}")
        summary_lines.append(f"  - Word clustering quality: {result['word_clustering_silhouette']:.4f}")
        summary_lines.append(f"  - Document clustering quality(silhouette): {result['doc_clustering_silhouette']:.4f}")
        summary_lines.append(f"  - Document clustering quality(Davies-Bouldin): {result['doc_clustering_davies']:.4f}")
    
    # Join all lines into a single string
    summary_text = "\n".join(summary_lines)
    
    # Print to console
    print(summary_text)
    
    # Save to TXT file
    with open("word2vec_bow_summary.txt", "w") as f:
        f.write(summary_text)
    
    print("\n✓ Summary saved to 'word2vec_bow_summary.txt'")

    
    return results

if __name__ == "__main__":
    results = main()
    
    import json
    with open('word2vec_bow_results.json', 'w') as f:
        json_results = []
        for r in results:
            json_result = {
                'doc_vector_dimension': int(r['doc_vector_dimension']),
                'word_vector_size': int(r['word_vector_size']),
                'n_word_clusters': int(r['n_word_clusters']),
                'word_clustering_silhouette': float(r['word_clustering_silhouette']),
                'doc_clustering_silhouette': float(r['doc_clustering_silhouette']),
                'cluster_summary': [
                    {
                        'cluster_id': int(s['cluster_id']),
                        'num_documents': int(s['num_documents']),
                        'top_words': [word for word, count in s['top_words'][:5]]
                    }
                    for s in r['cluster_stats']
                ]
            }
            json_results.append(json_result)
        json.dump(json_results, f, indent=2)
    
    print("\n✓ Results saved to 'word2vec_bow_results.json'")