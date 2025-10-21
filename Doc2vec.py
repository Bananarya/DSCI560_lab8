import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath='dataset.csv'):
    df = pd.read_csv(filepath)
    return df

def preprocess_text(text):
    if pd.isna(text):
        return []
    # Convert to lowercase and split
    return text.lower().split()

def train_doc2vec_model(documents, vector_size=100, min_count=2, epochs=50, dm=1):
    # Create tagged documents
    tagged_docs = [TaggedDocument(preprocess_text(doc), [i]) 
                   for i, doc in enumerate(documents)]
    
    # Initialize and train model
    model = Doc2Vec(
        vector_size=vector_size,
        min_count=min_count,
        epochs=epochs,
        dm=dm,
        workers=4,
        seed=42
    )
    
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Generate document vectors
    vectors = np.array([model.infer_vector(preprocess_text(doc)) 
                       for doc in documents])
    
    return model, vectors

def cluster_documents(vectors, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    
    # Calculate clustering metrics
    silhouette = silhouette_score(vectors, labels)
    davies_bouldin = davies_bouldin_score(vectors, labels)
    
    return labels, silhouette, davies_bouldin, kmeans

def analyze_clusters(df, labels, n_top_words=10):
    df['cluster'] = labels
    cluster_analysis = []
    
    for cluster_id in range(max(labels) + 1):
        cluster_docs = df[df['cluster'] == cluster_id]
        
        # Get all words from cluster documents
        all_words = []
        for msg in cluster_docs['message']:
            if pd.notna(msg):
                all_words.extend(preprocess_text(msg))
        
        # Find most common words
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(n_top_words)
        
        analysis = {
            'cluster_id': cluster_id,
            'size': len(cluster_docs),
            'percentage': len(cluster_docs) / len(df) * 100,
            'top_words': [word for word, _ in top_words],
            'sample_messages': cluster_docs['message'].head(3).tolist()
        }
        cluster_analysis.append(analysis)
    
    return cluster_analysis

def compare_configurations(df, configs):
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}: Vector Size = {config['vector_size']}")
        print('='*60)
        
        # Train Doc2Vec model
        print(f"Training Doc2Vec with parameters:")
        print(f"  - Vector size: {config['vector_size']}")
        print(f"  - Min count: {config.get('min_count', 2)}")
        print(f"  - Epochs: {config.get('epochs', 50)}")
        print(f"  - DM mode: {config.get('dm', 1)} {'(PV-DM)' if config.get('dm', 1) == 1 else '(PV-DBOW)'}")
        
        model, vectors = train_doc2vec_model(
            df['message'].tolist(),
            vector_size=config['vector_size'],
            min_count=config.get('min_count', 2),
            epochs=config.get('epochs', 50),
            dm=config.get('dm', 1)
        )
        
        # Cluster documents
        n_clusters = config.get('n_clusters', 5)
        print(f"Clustering documents into {n_clusters} clusters...")
        labels, silhouette, davies_bouldin, kmeans = cluster_documents(vectors, n_clusters)
        
        cluster_analysis = analyze_clusters(df.copy(), labels)
        
        result = {
            'config': config,
            'model': model,
            'vectors': vectors,
            'labels': labels,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'cluster_analysis': cluster_analysis,
            'inertia': kmeans.inertia_
        }
        results.append(result)
        
        print(f"\nClustering Metrics:")
        print(f"  - Silhouette Score: {silhouette:.4f} (higher is better, range: [-1, 1])")
        print(f"  - Davies-Bouldin Score: {davies_bouldin:.4f} (lower is better)")
        print(f"  - Inertia: {kmeans.inertia_:.2f}")
        
        print(f"\nCluster Distribution:")
        for analysis in cluster_analysis:
            print(f"  Cluster {analysis['cluster_id']}: {analysis['size']} documents ({analysis['percentage']:.1f}%)")
            print(f"    Top words: {', '.join(analysis['top_words'][:5])}")
    
    return results

def evaluate_best_configuration(results):
    print("\n" + "="*60)
    print("EVALUATION: BEST CONFIGURATION")
    print("="*60)
    
    # Sort by silhouette score (higher is better)
    sorted_by_silhouette = sorted(results, key=lambda x: x['silhouette_score'], reverse=True)
    
    print("\nRanking by Silhouette Score (higher is better):")
    for i, result in enumerate(sorted_by_silhouette):
        print(f"  {i+1}. Vector size {result['config']['vector_size']}: {result['silhouette_score']:.4f}")
    
    # Sort by Davies-Bouldin score (lower is better)
    sorted_by_db = sorted(results, key=lambda x: x['davies_bouldin_score'])
    
    print("\nRanking by Davies-Bouldin Score (lower is better):")
    for i, result in enumerate(sorted_by_db):
        print(f"  {i+1}. Vector size {result['config']['vector_size']}: {result['davies_bouldin_score']:.4f}")
    
    # Best overall (considering both metrics)
    best = sorted_by_silhouette[0]
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"  Vector size: {best['config']['vector_size']}")
    print(f"  Silhouette Score: {best['silhouette_score']:.4f}")
    print(f"  Davies-Bouldin Score: {best['davies_bouldin_score']:.4f}")
    
    return best

def main():
    df = load_data('dataset.csv')
    print(f"Loaded {len(df)} documents")
    
    # Define THREE different configurations as required
    configurations = [
        {
            'vector_size': 50,
            'min_count': 2,
            'epochs': 50,
            'dm': 1, 
            'n_clusters': 5
        },
        {
            'vector_size': 100,
            'min_count': 2,
            'epochs': 50,
            'dm': 1, 
            'n_clusters': 5
        },
        {
            'vector_size': 200,
            'min_count': 2,
            'epochs': 50,
            'dm': 1,  
            'n_clusters': 5
        }
    ]
    
    # Compare configurations
    results = compare_configurations(df, configurations)
    
    # Evaluate best configuration
    best_config = evaluate_best_configuration(results)
    
    # Additional analysis: Compare cluster coherence
    print("\n" + "="*60)
    print("QUALITATIVE ANALYSIS: CLUSTER COHERENCE")
    print("="*60)
    
    for result in results:
        print(f"\nVector size {result['config']['vector_size']}:")
        print("Sample clusters and their characteristics:")
        
        # Show first 2 clusters for brevity
        for analysis in result['cluster_analysis'][:2]:
            print(f"\n  Cluster {analysis['cluster_id']}:")
            print(f"    Size: {analysis['size']} documents")
            print(f"    Theme (based on top words): {', '.join(analysis['top_words'][:7])}")
            if analysis['sample_messages']:
                sample = analysis['sample_messages'][0]
                if pd.notna(sample):
                    print(f"    Sample: \"{sample[:100]}...\"")
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    with open('doc2vec_results_summary.txt', 'w') as f:
        f.write("DOC2VEC CONFIGURATION COMPARISON RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for result in results:
            f.write(f"Configuration: Vector Size = {result['config']['vector_size']}\n")
            f.write(f"  Silhouette Score: {result['silhouette_score']:.4f}\n")
            f.write(f"  Davies-Bouldin Score: {result['davies_bouldin_score']:.4f}\n")
            f.write(f"  Inertia: {result['inertia']:.2f}\n\n")
        
        f.write(f"\nBest Configuration: Vector Size = {best_config['config']['vector_size']}\n")
    
    print("✓ Results saved to 'doc2vec_results_summary.txt'")
    
    for result in results:
        vector_size = result['config']['vector_size']
        np.save(f'doc2vec_vectors_{vector_size}.npy', result['vectors'])
        print(f"✓ Vectors saved to 'doc2vec_vectors_{vector_size}.npy'")
    
    return results

if __name__ == "__main__":
    results = main()