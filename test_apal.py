#!/usr/bin/env python3
"""
Script sederhana untuk menjalankan APAL dengan data yang sudah diperbaiki
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def test_data_loading():
    """Test loading data untuk memastikan format sudah benar"""
    
    print("=== TESTING DATA LOADING ===")
    
    # Load data dengan pandas
    filepath = "d:\\Intan APAL\\string_interactions.tsv"
    print(f"Loading data from: {filepath}")
    
    try:
        # Baca file TSV
        df = pd.read_csv(filepath, sep='\t')
        
        # Bersihkan nama kolom (hapus prefix #)
        df.columns = df.columns.str.replace('#', '', regex=False)
        
        print(f"Data loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Cek kolom yang diperlukan
        required_cols = ['node1', 'node2']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"\nMissing columns: {missing_cols}")
            return False
        else:
            print(f"\nAll required columns present: {required_cols}")
            
        # Statistik data
        print(f"\nData Statistics:")
        print(f"Number of interactions: {len(df)}")
        print(f"Unique proteins (node1): {df['node1'].nunique()}")
        print(f"Unique proteins (node2): {df['node2'].nunique()}")
        
        # Buat graph sederhana untuk test
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(row['node1'], row['node2'])
        
        print(f"\nTest Graph:")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
        
        return True
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False

def run_simple_apal():
    """Jalankan APAL dengan implementasi sederhana"""
    
    print("\n=== RUNNING SIMPLE APAL IMPLEMENTATION ===")
    
    # Import class APAL
    from apal_breast_cancer import APALCommunityDetector
    
    # Inisialisasi detector
    detector = APALCommunityDetector(threshold=0.7)
    
    # Load data
    filepath = "string_interactions.tsv"
    graph = detector.load_data(filepath)
    
    if graph is None:
        print("Failed to load graph!")
        return
    
    # Jalankan APAL dengan threshold yang berbeda
    thresholds = [0.5, 0.6, 0.7, 0.8]
    results = {}
    
    for t in thresholds:
        print(f"\n--- Running APAL with threshold: {t} ---")
        
        try:
            communities = detector.apal_algorithm(t=t)
            analysis = detector.analyze_communities()
            
            results[t] = {
                'num_communities': len(communities),
                'modularity': analysis.get('modularity', 0),
                'coverage': analysis.get('coverage', 0),
                'overlapping_nodes': len(analysis.get('overlapping_nodes', {}))
            }
            
            print(f"Communities found: {len(communities)}")
            print(f"Modularity: {analysis.get('modularity', 0):.4f}")
            print(f"Coverage: {analysis.get('coverage', 0):.4f}")
            
        except Exception as e:
            print(f"Error running APAL with threshold {t}: {str(e)}")
            results[t] = {'error': str(e)}
    
    # Tampilkan summary
    print(f"\n=== SUMMARY RESULTS ===")
    for t, result in results.items():
        if 'error' not in result:
            print(f"Threshold {t}: {result['num_communities']} communities, "
                  f"modularity={result['modularity']:.4f}, "
                  f"coverage={result['coverage']:.4f}")
        else:
            print(f"Threshold {t}: Error - {result['error']}")
    
    # Pilih threshold terbaik
    valid_results = {t: r for t, r in results.items() if 'error' not in r}
    if valid_results:
        best_t = max(valid_results.keys(), key=lambda x: valid_results[x]['modularity'])
        print(f"\nBest threshold: {best_t} (modularity: {valid_results[best_t]['modularity']:.4f})")
        
        # Jalankan sekali lagi dengan threshold terbaik untuk analisis detail
        detector.apal_algorithm(t=best_t)
        detector.save_results(f'apal_results_threshold_{best_t}.txt')
        print(f"Detailed results saved to: apal_results_threshold_{best_t}.txt")

if __name__ == "__main__":
    print("DETEKSI KOMUNITAS OVERLAP MENGGUNAKAN APAL")
    print("PADA PROTEIN KANKER PAYUDARA")
    print("=" * 50)
    
    # Test loading data dulu
    if test_data_loading():
        print("\nData loading test PASSED!")
        
        # Jalankan APAL
        try:
            run_simple_apal()
        except Exception as e:
            print(f"\nError running APAL: {str(e)}")
            print("Please check the implementation and try again.")
    else:
        print("\nData loading test FAILED!")
        print("Please check the data file format.")