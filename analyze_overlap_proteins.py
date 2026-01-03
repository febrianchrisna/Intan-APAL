import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from apal_breast_cancer import APALDetector

def load_graph(filepath):
    print(f"📁 Loading data dari: {filepath}")
    
    df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.replace('#', '', regex=False)
    
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['node1'], row['node2'])
    
    print(f"✅ Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    return G

def analyze_overlap_proteins(G, communities, top_n=20):
    print("=" * 80)
    print("ANALISIS OVERLAP PROTEINS")
    print("=" * 80)
    
    # Hitung jumlah komunitas untuk setiap protein
    overlap_counts = {}
    for node in G.nodes():
        count = len([c for c in communities if node in c])
        if count > 1:  # Hanya protein yang ada di >1 komunitas
            overlap_counts[node] = count
    
    print(f"\n📊 Total overlap proteins: {len(overlap_counts)}")
    print(f"📊 Total communities: {len(communities)}")
    
    # Sort by overlap count
    top_overlap = sorted(overlap_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create DataFrame
    df_overlap = pd.DataFrame(top_overlap, columns=['Protein', 'Num_Communities'])
    
    # Database fungsi biologis untuk protein kanker payudara yang umum
    biological_functions = {
        # Key signaling & tumor suppressors
        'TP53': 'Tumor suppressor, regulator siklus sel & apoptosis',
        'PIK3CA': 'Katalitik subunit PI3K, jalur PI3K/AKT/mTOR signaling',
        'BRCA1': 'DNA repair (homologous recombination), tumor suppressor',
        'BRCA2': 'DNA repair (homologous recombination), tumor suppressor',
        'PTEN': 'Tumor suppressor, antagonis jalur PI3K/AKT',
        'AKT1': 'Serin/threonin kinase, jalur PI3K/AKT signaling',
        'ERBB2': 'Reseptor tirosin kinase (HER2), oncogene',
        'EGFR': 'Reseptor tirosin kinase, growth factor signaling',
        'ESR1': 'Reseptor estrogen α, hormone signaling',
        'PGR': 'Reseptor progesteron, hormone signaling',
        
        # Cell cycle regulators
        'CDK4': 'Cyclin-dependent kinase 4, regulator siklus sel G1/S',
        'CDK6': 'Cyclin-dependent kinase 6, regulator siklus sel G1/S',
        'CCND1': 'Cyclin D1, regulator siklus sel',
        'RB1': 'Retinoblastoma protein, tumor suppressor siklus sel',
        'CDKN2A': 'Cyclin-dependent kinase inhibitor 2A (p16), tumor suppressor',
        'CDKN1A': 'Cyclin-dependent kinase inhibitor 1A (p21), cell cycle arrest',
        
        # Apoptosis & stress response
        'BCL2': 'Anti-apoptotic protein, regulator kematian sel',
        'BAX': 'Pro-apoptotic protein, mitochondrial death pathway',
        'CASP3': 'Caspase 3, executioner apoptosis',
        'CASP8': 'Caspase 8, initiator apoptosis (extrinsic pathway)',
        'ATM': 'DNA damage response kinase, checkpoint activation',
        'CHEK2': 'Checkpoint kinase 2, DNA damage response',
        
        # Transcription factors
        'MYC': 'Transcription factor, oncogene, regulator proliferasi',
        'JUN': 'Transcription factor, komponen AP-1',
        'FOS': 'Transcription factor, komponen AP-1',
        'STAT3': 'Signal transducer & transcription activator',
        'NFKB1': 'NF-kappa-B, transcription factor inflamasi & survival',
        
        # Growth factor signaling
        'MTOR': 'mTOR kinase, regulator pertumbuhan & metabolisme sel',
        'IGF1R': 'Insulin-like growth factor 1 receptor',
        'VEGFA': 'Vascular endothelial growth factor, angiogenesis',
        'TGFB1': 'Transforming growth factor beta 1',
        
        # Chromatin & epigenetics
        'KMT2C': 'Histone-lysine N-methyltransferase, chromatin remodeling',
        'ARID1A': 'AT-rich interaction domain 1A, chromatin remodeling',
        'CREBBP': 'CREB-binding protein, histone acetyltransferase',
        'EP300': 'E1A-binding protein p300, histone acetyltransferase',
        
        # Ubiquitin pathway
        'MDM2': 'E3 ubiquitin ligase, negative regulator TP53',
        'FBXW7': 'F-box protein, substrate recognition untuk ubiquitination',
        
        # Metabolism
        'IDH1': 'Isocitrate dehydrogenase 1, metabolisme TCA cycle',
        
        # Migration & metastasis
        'CDH1': 'E-cadherin, cell adhesion, tumor suppressor',
        'CTNNB1': 'Beta-catenin, Wnt signaling pathway',
        
        # DNA repair
        'MLH1': 'MutL homolog 1, DNA mismatch repair',
        'MSH2': 'MutS homolog 2, DNA mismatch repair',
        'PALB2': 'Partner & localizer of BRCA2, DNA repair',
        
        # Protein kinases
        'MAP2K1': 'MEK1, MAPK/ERK kinase, MAPK signaling',
        'MAP2K4': 'MEK4, MAPK kinase, stress response',
        'MAPK1': 'ERK2, MAPK signaling pathway',
        'MAPK3': 'ERK1, MAPK signaling pathway',
        'SRC': 'Proto-oncogene tyrosine kinase',
        
        # Others
        'GATA3': 'Transcription factor, luminal breast cancer marker',
        'RUNX1': 'Transcription factor, hematopoiesis & breast development',
        'NF1': 'Neurofibromin, negative regulator RAS signaling',
        'NCOR1': 'Nuclear receptor corepressor 1',
    }
    
    # Tambahkan fungsi biologis
    df_overlap['Biological_Function'] = df_overlap['Protein'].apply(
        lambda x: biological_functions.get(x, 'Unknown - requires manual annotation')
    )
    
    # Identifikasi komunitas untuk setiap protein
    protein_communities = {}
    for protein in df_overlap['Protein']:
        comms = [i+1 for i, c in enumerate(communities) if protein in c]
        protein_communities[protein] = comms
    
    df_overlap['Community_IDs'] = df_overlap['Protein'].apply(
        lambda x: ', '.join(map(str, protein_communities.get(x, [])))
    )
    
    return df_overlap

def visualize_overlap_proteins(df_overlap, output_dir='d:\\7. Intan\\Intan APAL'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Top Overlap Proteins Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Limit to top 5 for visualization
    df_top5 = df_overlap.head(5)
    
    # 1. Number of Communities (horizontal bar)
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_top5)))
    bars = ax.barh(range(len(df_top5)), df_top5['Num_Communities'], color=colors, edgecolor='black')
    ax.set_yticks(range(len(df_top5)))
    ax.set_yticklabels(df_top5['Protein'], fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Communities', fontweight='bold', fontsize=12)
    ax.set_title('Top 5 Overlap Proteins by Community Count', fontweight='bold', fontsize=13)
    ax.grid(alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (protein, count) in enumerate(zip(df_top5['Protein'], df_top5['Num_Communities'])):
        ax.text(count + 0.1, i, str(count), va='center', fontweight='bold', fontsize=10)
    
    # 2. Summary table with biological function
    ax = axes[1]
    ax.axis('off')
    
    # Create table data
    table_data = []
    for i, row in df_top5.iterrows():
        # Truncate biological function if too long
        bio_func = row['Biological_Function']
        if len(bio_func) > 50:
            bio_func = bio_func[:47] + '...'
        table_data.append([
            row['Protein'],
            row['Num_Communities'],
            bio_func
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Protein', 'Communities', 'Fungsi Biologis'],
                    cellLoc='left',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    # Set column widths
    table.auto_set_column_width([0, 1, 2])
    
    ax.set_title('Top 5 Overlap Proteins Summary', fontweight='bold', fontsize=13, pad=20)
    
    plt.tight_layout()
    
    # Save figure
    output_path = f'{output_dir}\\top_overlap_proteins_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Visualisasi disimpan ke: {output_path}")
    plt.show()

def create_network_subgraph(G, communities, top_proteins, output_dir='d:\\7. Intan\\Intan APAL'):
    # Get all neighbors of top proteins
    all_nodes = set(top_proteins)
    for protein in top_proteins:
        all_nodes.update(G.neighbors(protein))
    
    # Limit size for visualization
    if len(all_nodes) > 100:
        # Keep only top proteins and their direct connections to other top proteins
        all_nodes = set(top_proteins)
        for p1 in top_proteins:
            for p2 in top_proteins:
                if p1 != p2 and G.has_edge(p1, p2):
                    all_nodes.add(p1)
                    all_nodes.add(p2)
    
    subgraph = G.subgraph(all_nodes)
    
    plt.figure(figsize=(16, 12))
    
    # Layout
    pos = nx.spring_layout(subgraph, k=2, iterations=100, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2, edge_color='gray', width=0.5)
    
    # Separate overlap proteins from others
    overlap_nodes = [n for n in subgraph.nodes() if n in top_proteins]
    other_nodes = [n for n in subgraph.nodes() if n not in top_proteins]
    
    # Draw non-overlap nodes (smaller, lighter)
    if other_nodes:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=other_nodes,
                              node_color='lightblue', node_size=150, 
                              alpha=0.5, edgecolors='gray', linewidths=0.5)
    
    # Draw overlap nodes (larger, colorful)
    if overlap_nodes:
        node_degrees = [subgraph.degree(n) for n in overlap_nodes]
        nx.draw_networkx_nodes(subgraph, pos, nodelist=overlap_nodes,
                              node_color='red', node_size=[d*50 for d in node_degrees],
                              alpha=0.9, edgecolors='darkred', linewidths=2)
    
    # Draw labels for overlap proteins only
    overlap_labels = {node: node for node in overlap_nodes}
    nx.draw_networkx_labels(subgraph, pos, overlap_labels, 
                           font_size=10, font_weight='bold', font_color='black')
    
    plt.title(f'Overlap Proteins Network (Top {len(top_proteins)} Proteins)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    output_path = f'{output_dir}\\overlap_proteins_network.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Network visualization disimpan ke: {output_path}")
    plt.show()

def main():
    # Configuration
    DATA_FILE = "d:\\7. Intan\\Intan APAL\\string_interactions.tsv"
    THRESHOLD = 0.3  # Use optimal threshold
    TOP_N = 5  # Number of top proteins to analyze
    OUTPUT_DIR = "d:\\7. Intan\\Intan APAL"
    
    print("=" * 80)
    print("ANALISIS TOP OVERLAP PROTEINS")
    print("=" * 80)
    print(f"\nData file: {DATA_FILE}")
    print(f"APAL Threshold: {THRESHOLD}")
    print(f"Top N proteins: {TOP_N}")
    
    # Load graph
    G = load_graph(DATA_FILE)
    
    # Run APAL
    print(f"\n🚀 Running APAL algorithm with threshold t={THRESHOLD}...")
    detector = APALDetector(G)
    communities = detector.apal_algorithm(t=THRESHOLD)
    print(f"✅ Detected {len(communities)} communities\n")
    
    # Analyze overlap proteins
    df_overlap = analyze_overlap_proteins(G, communities, top_n=TOP_N)
    
    # Print table
    print("\n" + "=" * 80)
    print(f"TOP {TOP_N} OVERLAP PROTEINS")
    print("=" * 80)
    print("\n" + df_overlap.to_string(index=False))
    
    # Print markdown table (for paper/documentation)
    print("\n" + "=" * 80)
    print("MARKDOWN TABLE (untuk dokumentasi/paper)")
    print("=" * 80)
    
    # Simplified table for markdown
    df_markdown = df_overlap[['Protein', 'Num_Communities', 'Biological_Function']].copy()
    df_markdown.columns = ['Protein', 'Communities', 'Fungsi Biologis']
    print("\n```markdown")
    print(df_markdown.head(10).to_markdown(index=False))
    print("```")
    
    # Save to CSV
    csv_path = f'{OUTPUT_DIR}\\top_overlap_proteins.csv'
    df_overlap.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Hasil disimpan ke: {csv_path}")
    
    # Create visualizations
    visualize_overlap_proteins(df_overlap, OUTPUT_DIR)
    
    # Create network visualization of overlap proteins
    top_proteins = df_overlap.head(5)['Protein'].tolist()
    create_network_subgraph(G, communities, top_proteins, OUTPUT_DIR)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("STATISTIK OVERLAP")
    print("=" * 80)
    print(f"\n📊 Total proteins dalam network: {G.number_of_nodes()}")
    print(f"📊 Total overlap proteins (>1 komunitas): {len(df_overlap)}")
    print(f"📊 Persentase overlap: {len(df_overlap)/G.number_of_nodes()*100:.2f}%")
    print(f"📊 Max communities per protein: {df_overlap['Num_Communities'].max()}")
    print(f"📊 Avg communities per overlap protein: {df_overlap['Num_Communities'].mean():.2f}")
    
    print("\n" + "=" * 80)
    print("ANALISIS SELESAI")
    print("=" * 80)

if __name__ == "__main__":
    main()
