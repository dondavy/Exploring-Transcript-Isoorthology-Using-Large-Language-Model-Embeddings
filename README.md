## Exploring Transcript Isoorthology Using Large Language Model Embeddings
---
<h2 align="center">A Computational Framework for Inferring Isoform Orthology via LLM Embeddings </h2>

<p align="center">
  <img src="./assets/overview-diagram.png" width="100%" title="Overview">
</p>

# Transcript-Homology-Clustering
A command-line tool designed to infer high-resolution homology relationships between transcripts by combining protein/gene embeddings with phylogenetic tree topology.

By leveraging unsupervised clustering (k-Medoids/Means) and Direct Coupling Analysis logic, this tool classifies transcript pairs into specific categories like ortho-isoorthologs, para-isoorthologs, and recent paralogs.

🚀 Key Features
Automated Clustering: Automatically selects the optimal number of clusters (k) using Silhouette Score heuristics and the principle of parsimony.

Evolutionary Context: Integrates NHX gene trees to distinguish between speciation and duplication events.

Embedding Support: Works with any 1D numpy-based embeddings (e.g., ProtBert, ESM-2).

Flexible Metrics: Supports both Cosine and Euclidean distance metrics for high-dimensional similarity.

🛠 Prerequisites
Environment Setup

You will need Python 3.8+ and the following libraries:

Bash
pip install numpy pandas scikit-learn scikit-learn-extra ete3 openpyxl
Required Input Files

Gene Tree (.nhx): A reconciled tree where internal nodes contain the ev attribute (e.g., ev=speciation or ev=duplication).

Source2Target (.tsv): A tab-separated file mapping transcript IDs to Gene IDs.

Embeddings Directory: A folder containing .npy files named after your transcript IDs.

💻 Usage
Run the analysis via src/main.py:

Bash
python src/main.py \
    --tree data/gene_tree.nhx \
    --source2target data/mapping.tsv \
    --embedding_path data/embeddings/ \
    --clustering_algorithm kMedoids \
    --distance_metric Cosine \
    --cluster_resolution_min 2 \
    --cluster_resolution_max 30 \
    --output_path ./output
Argument Reference

Argument	Default	Description
--tree	Required	Path to the NHX gene tree file.
--source2target	Required	Path to the Transcript-to-Gene TSV mapping.
--embedding_path	Required	Directory containing the .npy embedding files.
--clustering_algorithm	kMedoids	kMedoids (recommended for Cosine) or kMeans.
--distance_metric	Cosine	Cosine or Euclidean.
--cluster_resolution_min	2	Minimum k for the Silhouette heuristic.
--cluster_resolution_max	30	Maximum k for the Silhouette heuristic.
--output_path	.	Directory where the final Excel report will be saved.
🧬 Methodology
1. Optimal K Selection

The tool runs clustering across the specified range [k 
min
​	
 ,k 
max
​	
 ]. It calculates the Silhouette Score for each k.

To ensure the most biological result, it selects the smallest k that achieves a score within 0.02 of the global maximum. If the maximum score is below 0.3, the tool treats the data as a single cluster (unimodal).

2. Homology Inference Logic

The tool groups transcripts into clusters and identifies a Medoid (the most representative transcript) for each group. Relationships are then defined:

Primary Orthologs: Transcripts from different genes that are the "closest" to each other within a cluster.

Ortho-isoorthologs: Primary pairs linked via a Speciation event in the tree.

Para-isoorthologs: Primary pairs linked via a Duplication event in the tree.

Recent Paralogs: Transcripts from the same gene found within the same cluster.

📊 Output
The tool generates an ortholog_report.xlsx file.

Column	Description
tr_a / tr_b	The pair of transcript IDs being compared.
gene_a / gene_b	The corresponding Gene IDs from the mapping file.
relation	The specific biological relationship (e.g., ortho-isoorthologs).
type	The broad category (primary orthologs vs secondary orthologs).