## Exploring Transcript Isoorthology Using Large Language Model Embeddings
---

<p align="center">
  <img src="./assets/overview-diagram.png" width="100%" title="Overview">
</p>

<h3 align="center">A Computational Framework for Inferring Isoform Orthology via LLM Embeddings </h3>


A framework designed to infer high-resolution homology relationships between transcripts by combining protein/gene embeddings with gene homology relations.

By leveraging unsupervised clustering (kMedoids/kMeans), this tool classifies transcript pairs into specific categories like ortho-isoorthologs, para-isoorthologs, and recent paralogs.

## 🚀 Key Features
 * Automated Clustering: Automatically selects the optimal number of clusters (k) using Silhouette Score heuristics and the principle of parsimony.

 * Evolutionary Context: Integrates NHX gene trees to distinguish between speciation and duplication events.

 * Embedding Support: Works with any 1D numpy-based embeddings (e.g., ProtT5, ESM-2).

 * Flexible Metrics: Supports both Cosine and Euclidean distance metrics for high-dimensional similarity.

## 🛠 Prerequisites
#### Environment Setup

<div style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1000px;">
  
  <p>You will need <strong>Python 3.8+</strong> and the following libraries installed in your environment:</p>

  <div style="background-color: #282c34; border-radius: 6px; padding: 15px; margin: 15px 0; overflow-x: auto;">
    <code style="color: #abb2bf; font-family: 'Courier New', Courier, monospace; font-size: 14px;"><span style="color: #d19a66;">pip</span> install numpy pandas scikit-learn scikit-learn-extra ete3 openpyxl</code>
  </div>

  <h5 style="color: #2c3e50; margin-bottom: 10px;">Key Library Roles:</h5>
  <ul style="list-style-type: none; padding-left: 0;">
    <li style="margin-bottom: 8px; padding-left: 15px; border-left: 3px solid #3498db;">
      <strong>ete3:</strong> Used for parsing the NHX gene trees and identifying evolutionary events.
    </li>
    <li style="margin-bottom: 8px; padding-left: 15px; border-left: 3px solid #3498db;">
      <strong>scikit-learn-extra:</strong> Required specifically for the <code>kMedoids</code> algorithm.
    </li>
    <li style="margin-bottom: 8px; padding-left: 15px; border-left: 3px solid #3498db;">
      <strong>openpyxl:</strong> Necessary for exporting the final homology report to Excel format.
    </li>
  </ul>
</div>

#### Required Input Files

  * Gene Tree (.nhx): A reconciled tree where internal nodes contain the ev attribute (e.g., ev=speciation or ev=duplication).

  * Source2Target (.tsv): A tab-separated file mapping transcript IDs to Gene IDs.

  * Embeddings Directory: A folder containing .npy files named after your transcript IDs.

## 💻 Usage
  #### Run the analysis via src/main.py:

    
    python3 src/main.py \
        --tree data/example/ENSGT00390000000002.NHX \
        --source2target data/example/ENSGT00390000000002_source2target.tsv \
        --embedding_path ata/example/ENSGT00390000000002/ \
        --clustering_algorithm kMedoids \
        --distance_metric Cosine \
        --cluster_resolution_min 2 \
        --cluster_resolution_max 30 \
        --output_path data/example/kmedoids+cosine
    

  #### Argument Reference

  Argument	Default	Description

    <table style="width:100%; border-collapse: collapse; margin-top: 20px; font-family: Arial, sans-serif; font-size: 14px; line-height: 1.5;">
  <thead>
    <tr style="background-color: #2c3e50; color: #ffffff; text-align: left;">
      <th style="padding: 12px; border: 1px solid #ddd;">Argument</th>
      <th style="padding: 12px; border: 1px solid #ddd;">Default</th>
      <th style="padding: 12px; border: 1px solid #ddd;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 12px; border: 1px solid #ddd;"><code>--tree</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;"><strong>Required</strong></td>
      <td style="padding: 12px; border: 1px solid #ddd;">Path to the NHX gene tree file.</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;"><code>--source2target</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;"><strong>Required</strong></td>
      <td style="padding: 12px; border: 1px solid #ddd;">Path to the Transcript-to-Gene TSV mapping.</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 12px; border: 1px solid #ddd;"><code>--embedding_path</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;"><strong>Required</strong></td>
      <td style="padding: 12px; border: 1px solid #ddd;">Directory containing the <code>.npy</code> embedding files.</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;"><code>--clustering_algorithm</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;"><code>kMedoids</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;"><code>kMedoids</code> (recommended for Cosine) or <code>kMeans</code>.</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 12px; border: 1px solid #ddd;"><code>--distance_metric</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;"><code>Cosine</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;"><code>Cosine</code> or <code>Euclidean</code>.</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;"><code>--cluster_resolution_min</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;"><code>2</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;">Minimum <em>k</em> for the Silhouette heuristic.</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 12px; border: 1px solid #ddd;"><code>--cluster_resolution_max</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;"><code>30</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;">Maximum <em>k</em> for the Silhouette heuristic.</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;"><code>--output_path</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;"><code>.</code></td>
      <td style="padding: 12px; border: 1px solid #ddd;">Directory where the final Excel report will be saved.</td>
    </tr>
  </tbody>
</table>


  #### 🧬 Methodology
  
    1. Optimal K Selection

        The tool runs clustering across the specified range [kmin,kmax]. It calculates the Silhouette Score for each k.
        
        To ensure the most biological result, it selects the smallest k that achieves a score within 0.02 of the global maximum. If the maximum score is below 0.3, the tool treats the data as a single cluster (unimodal).

    2. Homology Inference Logic

        The tool groups transcripts into clusters and identifies a Medoid (the most representative transcript) for each group. Relationships are then defined:

        Primary Orthologs: Transcripts from different genes that are the "closest" to each other within a cluster.

        Ortho-isoorthologs: Primary pairs linked via a Speciation event in the tree.

        Para-isoorthologs: Primary pairs linked via a Duplication event in the tree.

        Recent Paralogs: Transcripts from the same gene found within the same cluster.



### 📊 Output
The tool generates an ortholog_report.xlsx file.

<table style="width:100%; border-collapse: collapse; margin-top: 20px; font-family: Arial, sans-serif;">
  <thead>
    <tr style="background-color: #f2f2f2; text-align: left;">
      <th style="padding: 12px; border: 1px solid #ddd;">Column</th>
      <th style="padding: 12px; border: 1px solid #ddd;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;"><strong>tr_a / tr_b</strong></td>
      <td style="padding: 12px; border: 1px solid #ddd;">The pair of transcript IDs being compared.</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;"><strong>gene_a / gene_b</strong></td>
      <td style="padding: 12px; border: 1px solid #ddd;">The corresponding Gene IDs from the mapping file.</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;"><strong>relation</strong></td>
      <td style="padding: 12px; border: 1px solid #ddd;">The specific biological relationship (e.g., <code>ortho-isoorthologs</code>).</td>
    </tr>
    <tr>
      <td style="padding: 12px; border: 1px solid #ddd;"><strong>type</strong></td>
      <td style="padding: 12px; border: 1px solid #ddd;">The broad category (<code>primary orthologs</code> vs <code>secondary orthologs</code>).</td>
    </tr>
  </tbody>
</table>