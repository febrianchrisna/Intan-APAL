import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from apal_breast_cancer import APALDetector
import time

def load_graph(filepath):
    print(f"📁 Loading data dari: {filepath}")
    
    df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.replace('#', '', regex=False)
    
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['node1'], row['node2'])
    
    print(f"✅ Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    return G

def run_threshold_experiment(G, thresholds):
    print("=" * 80)
    print("EKSPERIMEN THRESHOLD APAL")
    print("=" * 80)
    
    results = []
    
    for t in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing Threshold t = {t}")
        print(f"{'='*60}")
        
        # Inisialisasi detector
        detector = APALDetector(G)
        
        # Jalankan algoritma dan ukur waktu
        start_time = time.time()
        communities = detector.apal_algorithm(t=t)
        execution_time = time.time() - start_time
        
        # Analisis hasil
        analysis = detector.analyze_communities()
        
        # Hitung metrik tambahan
        community_sizes = [len(c) for c in communities]
        num_trivial = sum(1 for c in communities if len(c) <= 3)
        num_small = sum(1 for c in communities if len(c) <= 5)
        num_medium = sum(1 for c in communities if 6 <= len(c) <= 20)
        num_large = sum(1 for c in communities if len(c) > 20)
        
        # Normalized node cuts per community
        psi_values = [detector.normalized_node_cut(c) for c in communities]
        
        result = {
            'Threshold': t,
            'Communities': len(communities),
            'Avg_Size': round(np.mean(community_sizes), 2) if community_sizes else 0,
            'Min_Size': min(community_sizes) if community_sizes else 0,
            'Max_Size': max(community_sizes) if community_sizes else 0,
            'Avg_Psi': round(analysis.get('avg_normalized_node_cut', 0), 4),
            'Min_Psi': round(min(psi_values), 4) if psi_values else 0,
            'Max_Psi': round(max(psi_values), 4) if psi_values else 0,
            'Trivial_<=3': num_trivial,
            'Small_<=5': num_small,
            'Medium_6-20': num_medium,
            'Large_>20': num_large,
            'Overlap_Nodes': analysis.get('num_overlapping_nodes', 0),
            'Coverage': round(analysis.get('coverage', 0), 4),
            'Exec_Time_s': round(execution_time, 3)
        }
        
        results.append(result)
        
        # Print summary
        print(f"\n📊 HASIL:")
        print(f"   Komunitas terdeteksi: {result['Communities']}")
        print(f"   Ukuran rata-rata: {result['Avg_Size']:.2f}")
        print(f"   Normalized Node Cut (Ψ) rata-rata: {result['Avg_Psi']:.4f}")
        print(f"   Overlap protein: {result['Overlap_Nodes']}")
        print(f"   Coverage: {result['Coverage']:.4f}")
        print(f"   Waktu eksekusi: {result['Exec_Time_s']:.3f}s")
    
    return pd.DataFrame(results)

def visualize_threshold_comparison(df_results, output_dir='d:\\7. Intan\\Intan APAL'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Threshold Experiment Results - APAL Algorithm', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    thresholds = df_results['Threshold']
    
    # 1. Number of Communities
    ax = axes[0, 0]
    ax.plot(thresholds, df_results['Communities'], 'o-', linewidth=2, markersize=8, color='#6366f1')
    ax.set_xlabel('Threshold (t)', fontweight='bold')
    ax.set_ylabel('Number of Communities', fontweight='bold')
    ax.set_title('Communities vs Threshold')
    ax.grid(alpha=0.3)
    ax.axvline(0.3, color='red', linestyle='--', alpha=0.5, label='t=0.3')
    ax.legend()
    
    # 2. Average Normalized Node Cut (Ψ)
    ax = axes[0, 1]
    ax.plot(thresholds, df_results['Avg_Psi'], 'o-', linewidth=2, markersize=8, color='#06b6d4')
    ax.set_xlabel('Threshold (t)', fontweight='bold')
    ax.set_ylabel('Avg Normalized Node Cut (Ψ)', fontweight='bold')
    ax.set_title('Quality Metric vs Threshold\n(Lower is Better)')
    ax.grid(alpha=0.3)
    ax.axvline(0.3, color='red', linestyle='--', alpha=0.5, label='t=0.3')
    ax.legend()
    
    # 3. Average Community Size
    ax = axes[0, 2]
    ax.plot(thresholds, df_results['Avg_Size'], 'o-', linewidth=2, markersize=8, color='#10b981')
    ax.set_xlabel('Threshold (t)', fontweight='bold')
    ax.set_ylabel('Average Community Size', fontweight='bold')
    ax.set_title('Avg Size vs Threshold')
    ax.grid(alpha=0.3)
    ax.axvline(0.3, color='red', linestyle='--', alpha=0.5, label='t=0.3')
    ax.legend()
    
    # 4. Community Size Distribution
    ax = axes[1, 0]
    x = np.arange(len(thresholds))
    width = 0.2
    ax.bar(x - width*1.5, df_results['Trivial_<=3'], width, label='Trivial (≤3)', color='#ef4444')
    ax.bar(x - width*0.5, df_results['Small_<=5'], width, label='Small (≤5)', color='#f59e0b')
    ax.bar(x + width*0.5, df_results['Medium_6-20'], width, label='Medium (6-20)', color='#10b981')
    ax.bar(x + width*1.5, df_results['Large_>20'], width, label='Large (>20)', color='#3b82f6')
    ax.set_xlabel('Threshold (t)', fontweight='bold')
    ax.set_ylabel('Number of Communities', fontweight='bold')
    ax.set_title('Size Distribution by Threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 5. Coverage & Overlap
    ax = axes[1, 1]
    ax2 = ax.twinx()
    line1 = ax.plot(thresholds, df_results['Coverage'], 'o-', linewidth=2, 
                    markersize=8, color='#8b5cf6', label='Coverage')
    line2 = ax2.plot(thresholds, df_results['Overlap_Nodes'], 's--', linewidth=2, 
                     markersize=8, color='#ec4899', label='Overlap Nodes')
    ax.set_xlabel('Threshold (t)', fontweight='bold')
    ax.set_ylabel('Coverage', fontweight='bold', color='#8b5cf6')
    ax2.set_ylabel('Overlap Nodes', fontweight='bold', color='#ec4899')
    ax.set_title('Coverage & Overlap vs Threshold')
    ax.grid(alpha=0.3)
    ax.axvline(0.3, color='red', linestyle='--', alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    
    # 6. Execution Time
    ax = axes[1, 2]
    ax.plot(thresholds, df_results['Exec_Time_s'], 'o-', linewidth=2, markersize=8, color='#f97316')
    ax.set_xlabel('Threshold (t)', fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontweight='bold')
    ax.set_title('Computational Cost vs Threshold')
    ax.grid(alpha=0.3)
    ax.axvline(0.3, color='red', linestyle='--', alpha=0.5, label='t=0.3')
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = f'{output_dir}\\threshold_experiment_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Visualisasi disimpan ke: {output_path}")
    plt.show()

def print_recommendation(df_results):
    print("\n" + "=" * 80)
    print("REKOMENDASI THRESHOLD")
    print("=" * 80)
    
    # Find optimal threshold based on multiple criteria
    # Kriteria: Psi rendah, coverage tinggi, ukuran komunitas reasonable
    
    # Normalize metrics (0-1)
    df_norm = df_results.copy()
    df_norm['Psi_norm'] = 1 - (df_norm['Avg_Psi'] - df_norm['Avg_Psi'].min()) / (df_norm['Avg_Psi'].max() - df_norm['Avg_Psi'].min() + 1e-10)
    df_norm['Coverage_norm'] = df_norm['Coverage']
    df_norm['Size_norm'] = 1 - abs(df_norm['Avg_Size'] - 10) / 20  # Optimal size around 10
    
    # Combined score (weighted)
    df_norm['Score'] = (
        0.4 * df_norm['Psi_norm'] +        # 40% weight on quality
        0.3 * df_norm['Coverage_norm'] +   # 30% weight on coverage
        0.2 * df_norm['Size_norm'] +       # 20% weight on size
        0.1 * (1 - df_norm['Trivial_<=3'] / df_norm['Communities'].max())  # 10% penalty for trivial
    )
    
    best_idx = df_norm['Score'].idxmax()
    best_threshold = df_results.loc[best_idx, 'Threshold']
    
    print(f"\n🏆 THRESHOLD OPTIMAL: t = {best_threshold}")
    print(f"\nAlasan:")
    print(f"  ✓ Normalized Node Cut (Ψ): {df_results.loc[best_idx, 'Avg_Psi']:.4f} (semakin rendah semakin baik)")
    print(f"  ✓ Coverage: {df_results.loc[best_idx, 'Coverage']:.4f} (proporsi node yang tercakup)")
    print(f"  ✓ Ukuran komunitas rata-rata: {df_results.loc[best_idx, 'Avg_Size']:.2f} protein")
    print(f"  ✓ Total komunitas: {df_results.loc[best_idx, 'Communities']}")
    print(f"  ✓ Overlap protein: {df_results.loc[best_idx, 'Overlap_Nodes']}")
    
    print(f"\n📋 Perbandingan dengan threshold lain:")
    for _, row in df_results.iterrows():
        if row['Threshold'] == best_threshold:
            print(f"  → t={row['Threshold']:.1f}: Ψ={row['Avg_Psi']:.4f}, Communities={row['Communities']}, Coverage={row['Coverage']:.4f} ⭐ OPTIMAL")
        else:
            print(f"    t={row['Threshold']:.1f}: Ψ={row['Avg_Psi']:.4f}, Communities={row['Communities']}, Coverage={row['Coverage']:.4f}")

def main():
    # Configuration
    DATA_FILE = "d:\\7. Intan\\Intan APAL\\string_interactions.tsv"  # Gunakan file yang lebih kecil untuk testing cepat
    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
    OUTPUT_DIR = "d:\\7. Intan\\Intan APAL"
    
    print("=" * 80)
    print("EKSPERIMEN THRESHOLD APAL - JUSTIFIKASI PEMILIHAN THRESHOLD OPTIMAL")
    print("=" * 80)
    print(f"\nData file: {DATA_FILE}")
    print(f"Thresholds to test: {THRESHOLDS}")
    
    # Load graph
    G = load_graph(DATA_FILE)
    
    # Run experiment
    df_results = run_threshold_experiment(G, THRESHOLDS)
    
    # Print results table
    print("\n" + "=" * 80)
    print("TABEL HASIL EKSPERIMEN")
    print("=" * 80)
    print("\n" + df_results.to_string(index=False))
    
    # Save results to CSV
    csv_path = f'{OUTPUT_DIR}\\threshold_experiment_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n💾 Hasil disimpan ke: {csv_path}")
    
    # Create visualization
    visualize_threshold_comparison(df_results, OUTPUT_DIR)
    
    # Print recommendation
    print_recommendation(df_results)
    
    print("\n" + "=" * 80)
    print("EKSPERIMEN SELESAI")
    print("=" * 80)

if __name__ == "__main__":
    main()
