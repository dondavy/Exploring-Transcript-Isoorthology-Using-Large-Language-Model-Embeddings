# src/main.py

# LIBRAIRIES

import sys
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min, pairwise_distances
from sklearn_extra.cluster import KMedoids  # Requires: pip install scikit-learn-extra
import csv
from ete3 import Tree
from itertools import combinations

# PARSE ARGS

def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="Process gene trees...")

    parser.add_argument("--tree", type=str, required=True, help="Path to gene tree")
    parser.add_argument("--source2target", type=str, required=True, help="Path to TSV")
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to embeddings")
    parser.add_argument("--clustering_algorithm", type=str, default="kMedoids", required=False, help="kMeans | KMedoids")
    parser.add_argument("--distance_metric", type=str, default="Cosine", required=False, help="Cosine | Euclidean")
    parser.add_argument("--cluster_resolution_max", type=int, default="30", required=False, help="Max K value. The script will test K from k_min up to this number.")
    parser.add_argument("--cluster_resolution_min", type=int, default="2", required=False, help="Max K value. The script will test K from this number up to k_max.")
    parser.add_argument("--output_path", type=str, default=".", required=False, help="output path (default='.')")


    args = parser.parse_args(args_list)
    # Validation
    if not os.path.exists(args.tree):
        parser.error(f"The file {args.tree} does not exist.")
    if not os.path.exists(args.source2target):
        parser.error(f"The file {args.source2target} does not exist.")
    if not os.path.isdir(args.embedding_path):
        parser.error(f"The directory {args.embedding_path} does not exist.")
    if not os.path.exists(args.output_path):
        parser.error(f"The output path {args.output_path} does not exist.")
    
    algo = args.clustering_algorithm.lower()
    metric = args.distance_metric.lower()

    # 2. Check forbidden combination
    if algo == "kmeans" and metric == "cosine":
        parser.error(
            "Invalid Configuration: kMeans does not support Cosine distance. "
            "Please use Euclidean with kMeans, or switch to kMedoids."
        )
    
    # 3. (Optional) Check for valid options generally
    valid_algos = ["kmeans", "kmedoids"]
    valid_metrics = ["cosine", "euclidean"]

    if algo not in valid_algos:
        parser.error(f"Unknown algorithm: {args.clustering_algorithm}")
    
    if metric not in valid_metrics:
        parser.error(f"Unknown metric: {args.distance_metric}")

    return args

# UTILS

def get_evol_event(tree, geneA, geneB):
    """
    Retrieve the evolutionary event (speciation vs. duplication) for a gene pair.

    Identifies the most recent common ancestor (MRCA) of two genes within a 
    reconciled phylogenetic tree and extracts the event type (ev) assigned to that node.

    Args:
        tree (ete3.TreeNode): A phylogenetic tree object, typically reconciled 
            with species information to identify evolutionary events (ev).
        gene_a (str): The identifier for the first gene/leaf in the tree.
        gene_b (str): The identifier for the second gene/leaf in the tree.

    Returns:
        str: The evolutionary event label. Common return values include:
            - 'speciation': Speciation (leads to orthologs)
            - 'duplication': Duplication (leads to paralogs)
            - 'None': If the event is undefined or nodes are missing.
    """
    return tree.get_common_ancestor([geneA, geneB]).ev

def read_npy(folder_npy):
    """
    Reads all .npy files in a folder and returns a DataFrame.
    
    Index: Filename (without .npy extension)
    Columns: emb_dim_0, emb_dim_1, ...
    """
    data_list = []
    index_list = []
    
    # Get all .npy files (sorted ensures reproducible order)
    files = sorted([f for f in os.listdir(folder_npy) if f.endswith('.npy')])
    
    if not files:
        raise ValueError(f"No .npy files found in {folder_npy}")

    # Loop through files and load data
    for filename in files:
        file_path = os.path.join(folder_npy, filename)
        
        try:
            # Load the numpy array
            embedding = np.load(file_path, allow_pickle=True)
            
            # Ensure it's 1D (flatten if it's 2D like [[...]])
            if embedding.ndim > 1:
                embedding = embedding.flatten()
                
            data_list.append(embedding)
            
            # Use filename without extension as the index
            name_without_ext = os.path.splitext(filename)[0].split('_')[0]
            index_list.append(name_without_ext)
            
        except Exception as e:
            raise ValueError(f"Error loading {filename}: {e}")

    # Create DataFrame
    df = pd.DataFrame(data_list, index=index_list)
    
    # Rename columns to emb_dim_0, emb_dim_1, ...
    # We do this dynamically based on the embedding size
    num_dims = df.shape[1]
    df.columns = [f"emb_dim_{i}" for i in range(1,num_dims+1)]
    
    return df

def read_s2t(file_path):
    """
    Reads a TSV file mapping transcript IDs to gene IDs.
    
    Args:
        file_path (str): Path to the TSV file.
        
    Returns:
        dict: {transcript_id: gene_id}
    """
    s2t_dict = {}
    
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            # Using csv.reader with delimiter='\t' for TSV files
            reader = csv.reader(f, delimiter='\t')
            
            for row in reader:
                # Skip empty rows or malformed lines
                if not row or len(row) < 2:
                    continue
                
                transcript_id = row[0].strip()
                gene_id = row[1].strip()
                
                s2t_dict[transcript_id] = gene_id
                
        print(f"Successfully loaded {len(s2t_dict)} transcript-to-gene mappings.")
        return s2t_dict

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the mapping file: {e}")
        return {}
    
def compute_clustering(df, k_min=2, k_max=10, algorithm='kMedoids', metric='cosine'):
    """
    Runs clustering for a range of k values and computes Silhouette Scores.
    
    Args:
        df (pd.DataFrame): Dataframe where rows are samples and columns are embedding dimensions.
        k_min (int): Minimum number of clusters to test (default 2).
        k_max (int): Maximum number of clusters to test (default 10).
        algorithm (str): 'kMeans' or 'kMedoids'.
        metric (str): 'euclidean' or 'cosine'.
        
    Returns:
        pd.DataFrame: A summary dataframe with columns ['k', 'silhouette_score'].
    """
    
    # 1. Input Validation
    # We enforce the constraint: KMeans only works well with Euclidean in this implementation.
    if algorithm.lower() == 'kmeans' and metric.lower() == 'cosine':
        raise ValueError("KMeans with Cosine distance is not supported. Use KMedoids or switch to Euclidean.")
    metric = metric.lower()
    
    results = []
    
    # Dictionary to store labels so we don't have to re-run the model later
    # Key = k, Value = array of labels
    cached_labels = {}
    
    # 2. Iterate through range of k
    # We go from k_min up to (and including) k_max
    print(f"Running {algorithm} ({metric}) for k={k_min} to {k_max}...")
    
    for k in range(k_min, k_max + 1):
        try:
            # Choose Algorithm ---
            if algorithm.lower() == 'kmeans':
                # KMeans always uses Euclidean distance
                model = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
                labels = model.fit_predict(df)
                
            elif algorithm.lower() == 'kmedoids':
                # KMedoids supports 'cosine' and 'euclidean' directly
                model = KMedoids(n_clusters=k, metric=metric, random_state=42, method='pam', init='k-medoids++')
                labels = model.fit_predict(df)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Compute Silhouette Score ---
            ## The score ranges from -1 to 1. 
            ## 1 = Well separated clusters
            ## 0 = Overlapping clusters
            ## -1 = Incorrect clustering
            score = silhouette_score(df, labels, metric=metric)
            
            results.append({'k': k, 'silhouette_score': score})
            print(f"  k={k}: Silhouette Score = {score:.4f}")
            cached_labels[k] = (model, labels)
            
        except Exception as e:
            print(f"Skipping k={k} due to error: {e}")

    # Convert to DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    
    # Find the best k (Heuristic Selection)
    best_k = 1
    best_labels = np.zeros(len(df), dtype=int)
    exemplar_ids = []
    
    if not results_df.empty:
        # Step A: Find the global maximum score
        max_score = results_df['silhouette_score'].max()
        
        # Step B: Check if the clustering is good enough
        if max_score < 0.3:
            print(f"\n⚠️ Max Silhouette Score ({max_score:.4f}) is < 0.3. The structure is weak and seems to be unimodal.")
            print("   -> Returning best_k = 1 (Entire set considered as one cluster).")
            best_k = 1
        else:
            # Step C: Find candidates within the "tolerance" range (max - 0.02)
            threshold = max_score - 0.02
            candidates = results_df[results_df['silhouette_score'] >= threshold]
            
            # Step D: Select the candidate with the MINIMUM k (Parsimony principle)
            # candidates is a DataFrame, we sort by 'k' and take the top one
            best_row = candidates.sort_values('k').iloc[0]
            
            best_k = int(best_row['k'])
            selected_score = best_row['silhouette_score']
            
            print(f"\n✅ Best k found: {best_k} (Score: {selected_score:.4f})")
            print(f"   (Selected smallest k with score >= {threshold:.4f})")
            
            model, best_labels = cached_labels[best_k]
            
            
            # 3. Find Exemplars (Actual Transcripts)
            if algorithm.lower() == 'kmeans':
                # Find the index of the point closest to each cluster center
                # pairwise_distances_argmin_min returns the index of the closest row in 'df'
                closest_indices, _ = pairwise_distances_argmin_min(model.cluster_centers_, df)
                exemplar_ids = df.index[closest_indices].tolist()
            else:
                # KMedoids already identifies real points as centers
                exemplar_indices = model.medoid_indices_
                exemplar_ids = df.index[exemplar_indices].tolist()

    # You might want to return best_k now, or attach it to the dataframe
    return results_df, best_k, best_labels, exemplar_ids

def compute_homology(data, labels, medoids, metric, tree_path, s2t):
    """
    Infer transcript homology relations based on distance metrics and phylogenetic constraints.

    Args:
        data : Feature matrix representing transcripts.
        labels (list): Unique identifiers corresponding to the rows in data.
        medoids (list): Centroid-like representatives used for cluster-based comparison.
        metric (str): The distance metric for similarity (e.g., 'euclidean' or 'cosine').
        tree_path (str): File path to the guide tree (Nexus format).
        s2t (dict): Mapping dictionary (e.g., Transcript-to-Gene) for ID resolution.

    Returns:
        pd.DataFrame: A table of homology relations containing:
            - 'tr_a/b': Source and target transcript IDs.
            - 'gene_a/b': Associated gene symbols for each transcript.
            - 'relation': Biological relationship (e.g., 'ortho-isoorthologs').
            - 'type': Classification of the match (e.g., 'primary orthologs').
    """
    metric = metric.lower()
    final_pairs = []
    # Load the Tree (NHX format supports 'ev' attribute for Duplication and Speciation)
    try:
        t = Tree(tree_path, format=1) 
    except Exception as e:
        print(f"Error loading tree: {e}")
        return None

    # Map Cluster results into a readable format
    # Index of 'data' is the transcript ID
    results = pd.DataFrame({
        'transcript': data.index,
        'cluster': labels,
        'gene': [s2t.get(tr, "Unknown") for tr in data.index]
    })
    
    # Analyze Cluster Separations
    
    
    for cluster_id in sorted(results['cluster'].unique()):
        cluster_members = results[results['cluster'] == cluster_id]
        cluster_transcripts = cluster_members['transcript']
        if len(cluster_transcripts) == 1:
            final_pairs.append({
                    'tr_a': cluster_transcripts.values[0],
                    'gene_a': s2t.get(cluster_transcripts.values[0], "Unknown"),
                    'tr_b': None,
                    'gene_b': None,
                    'relation': None,
                    'type': None
                })
            continue
        
        # Find the representative medoid for this cluster
        medoid_tr = medoids[cluster_id]
        medoid_vector = data.loc[[medoid_tr]].values
        distances = pairwise_distances(medoid_vector, data.loc[cluster_transcripts].values, metric=metric).flatten()

        temp_df = pd.DataFrame({
            'transcript': cluster_transcripts,
            'distance': distances,
            'gene': [s2t.get(tr) for tr in cluster_transcripts]
        })
        medoid_gene = s2t.get(medoid_tr, "Unknown")
        
        try:
            # Primary orthologs
            temp_df = temp_df[temp_df['gene'] != medoid_gene]
            unique_gene_results = (
                temp_df.sort_values('distance')
                    .groupby('gene')
                    .head(1) # Keep only the closest transcript per gene
            )
            
            unique_pool = unique_gene_results['transcript'].values
            
                
            # Ortho-isoorthologs and Para-isoorthologs
            for tr_a, tr_b in combinations(unique_pool, 2):
                gene_a = s2t.get(tr_a, "Unknown")
                gene_b = s2t.get(tr_b, "Unknown")
                ev = get_evol_event(t, gene_a, gene_b) 
                if ev == "speciation":
                    final_pairs.append({
                        'tr_a': tr_a,
                        'gene_a': gene_a,
                        'tr_b': tr_b,
                        'gene_b': gene_b,
                        'relation': 'ortho-isoorthologs',
                        'type': 'primary orthologs'
                    })
                elif ev == "duplication":
                    final_pairs.append({
                        'tr_a': tr_a,
                        'gene_a': gene_a,
                        'tr_b': tr_b,
                        'gene_b': gene_b,
                        'relation': 'para-isoorthologs',
                        'type': 'primary orthologs'
                    })
               
                
            # Recent paralogs
            already_rp_sets = []
            for tr in unique_pool:
                for tr_b in cluster_transcripts:
                    if tr != tr_b and (tr, tr_b) not in already_rp_sets and tr_b not in unique_pool:
                        gene_tr = s2t.get(tr, "Unknown")
                        gene_trb = s2t.get(tr_b, "Unknown")
                        if gene_tr == gene_trb:
                            final_pairs.append({
                                'tr_a': tr,
                                'gene_a': gene_tr,
                                'tr_b': tr_b,
                                'gene_b': gene_trb,
                                'relation': 'recent-paralogs',
                                'type': 'paralogs'
                            })
                            already_rp_sets.append((tr, tr_b))
                        else:
                            ev = get_evol_event(t, gene_tr, gene_trb) 
                            if ev == "speciation":
                                final_pairs.append({
                                    'tr_a': tr,
                                    'gene_a': gene_tr,
                                    'tr_b': tr_b,
                                    'gene_b': gene_trb,
                                    'relation': 'ortho-orthologs',
                                    'type': 'secondary orthologs'
                                })
                            elif ev == "duplication":
                                final_pairs.append({
                                    'tr_a': tr,
                                    'gene_a': gene_tr,
                                    'tr_b': tr_b,
                                    'gene_b': gene_trb,
                                    'relation': 'para-orthologs',
                                    'type': 'secondary orthologs'
                                })
                            already_rp_sets.append((tr, tr_b))
                    
            df_final_pairs = pd.DataFrame(final_pairs)
            return df_final_pairs
            
        except:
            raise ValueError("Something wrong with the inference step!")

# MAIN FUNCTION

def main(args):
    # Load Data
    data = read_npy(args.embedding_path)
    s2t = read_s2t(args.source2target)
    
    # Run Clustering
    results_df, best_k, best_labels, medoids = compute_clustering(
        data, 
        k_min=args.cluster_resolution_min,                        # Silhouette needs min 2 clusters
        k_max=args.cluster_resolution_max,  # Driven by your new argument
        algorithm=args.clustering_algorithm,
        metric=args.distance_metric
    )

    # Inferring transcript homology
    homology_df = compute_homology(
        data,
        labels=best_labels,
        medoids=medoids,
        metric=args.distance_metric,
        tree_path=args.tree,
        s2t=s2t
    )
    
    # Save Results
    homology_df['type'] = homology_df['type'].astype(str).str.strip()
    homology_df['relation'] = homology_df['relation'].astype(str).str.strip()
    homology_df = homology_df.sort_values(['type', 'relation'])
    df_grouped = homology_df.set_index(['type', 'relation'])
    df_grouped.to_excel(os.path.join(args.output_path,'ortholog_report.xlsx'))
    
    
    return True

if __name__ == "__main__":
    args = parse_args()
    main(args)