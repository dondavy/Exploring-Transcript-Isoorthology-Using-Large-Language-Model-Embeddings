import pytest
import argparse
from unittest.mock import patch
from src.main import parse_args, main # Import parse_args, NOT main!


# Kmedoids+cosine
@patch('os.path.exists', return_value=True)
@patch('os.path.isdir', return_value=True)
def test_parse_args_valid_1(mock_isdir, mock_exists):
    # Define the input
    input_tree = "data/example/ENSGT00390000000002.NHX"
    input_map  = "data/example/ENSGT00390000000002_source2target.tsv"
    input_emb  = "data/example/ENSGT00390000000002"
    input_clustering_algo = "kMedoids"
    input_metric = "Cosine"
    input_k_range_min = "3"
    input_k_range_max = "20"
    input_outpath = "data/example/kmedoids+cosine"
    
    test_args = [
        "--tree", input_tree,
        "--source2target", input_map,
        "--embedding_path", input_emb,
        "--clustering_algorithm", input_clustering_algo,
        "--distance_metric", input_metric,
        "--cluster_resolution_min", input_k_range_min,
        "--cluster_resolution_max", input_k_range_max,
        "--output_path", input_outpath
    ]
    
    # Call PARSE_ARGS 
    args = parse_args(test_args)
    
    # 3. Assertions (Must match the input variables exactly)
    assert args.tree == input_tree
    assert args.source2target == input_map
    assert args.embedding_path == input_emb
    
    # Call MAIN
    assert main(args) == True
    
# Kmedoids+euclidean
def test_parse_args_valid_2():
    # Define the input
    input_tree = "data/example/ENSGT00390000000002.NHX"
    input_map  = "data/example/ENSGT00390000000002_source2target.tsv"
    input_emb  = "data/example/ENSGT00390000000002"
    input_clustering_algo = "kMedoids"
    input_metric = "euclidean"
    input_k_range_min = "3"
    input_k_range_max = "10"
    input_outpath = "data/example/kmedoids+euclidean"
    
    test_args = [
        "--tree", input_tree,
        "--source2target", input_map,
        "--embedding_path", input_emb,
        "--clustering_algorithm", input_clustering_algo,
        "--distance_metric", input_metric,
        "--cluster_resolution_min", input_k_range_min,
        "--cluster_resolution_max", input_k_range_max,
        "--output_path", input_outpath
    ]
    
    # Call PARSE_ARGS 
    args = parse_args(test_args)
    
    # 3. Assertions (Must match the input variables exactly)
    assert args.tree == input_tree
    assert args.source2target == input_map
    assert args.embedding_path == input_emb
    
    # Call MAIN
    assert main(args) == True
    
    
# Kmeans+euclidean
def test_parse_args_valid_3():
    # Define the input
    input_tree = "data/example/ENSGT00390000000002.NHX"
    input_map  = "data/example/ENSGT00390000000002_source2target.tsv"
    input_emb  = "data/example/ENSGT00390000000002"
    input_clustering_algo = "kMeans"
    input_metric = "euclidean"
    input_k_range_min = "30"
    input_k_range_max = "35"
    input_outpath = "data/example/kmeans+euclidean"
    
    test_args = [
        "--tree", input_tree,
        "--source2target", input_map,
        "--embedding_path", input_emb,
        "--clustering_algorithm", input_clustering_algo,
        "--distance_metric", input_metric,
        "--cluster_resolution_min", input_k_range_min,
        "--cluster_resolution_max", input_k_range_max,
        "--output_path", input_outpath
    ]
    
    # Call PARSE_ARGS 
    args = parse_args(test_args)
    
    # 3. Assertions (Must match the input variables exactly)
    assert args.tree == input_tree
    assert args.source2target == input_map
    assert args.embedding_path == input_emb
    
    # Call MAIN
    assert main(args) == True
    
