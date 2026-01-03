import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Set, Iterable, Dict
import time
import warnings
warnings.filterwarnings('ignore')

class APALDetector:
    def __init__(self, graph):
        if not isinstance(graph, nx.Graph):
            raise TypeError("graph must be a networkx.Graph")
        
        # Pastikan graf sederhana (no self-loops, unweighted untuk algoritma)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(graph.nodes())
        # Buang self-loops
        self.graph.add_edges_from((u, v) for u, v in graph.edges() if u != v)
        
        self.communities: List[Set] = []
        self.node_to_communities = defaultdict(set)
        
        # Untuk tracking waktu eksekusi kumulatif
        self.execution_timeline = []  # List of (node_index, cumulative_time, communities_count)
    
    def get_neighbors(self, node) -> Set:
        return set(self.graph.neighbors(node))
    
    def intraconnectivity(self, community: Iterable) -> float:
        community = set(community)
        n = len(community)
        if n <= 1:
            return 0.0
        
        k = 0  # Sum of internal degrees
        for node in community:
            neighbors = self.get_neighbors(node)
            k += len(neighbors.intersection(community))
        
        max_possible = n * (n - 1)
        if max_possible == 0:
            return 0.0
        
        return k / max_possible
    
    @staticmethod
    def jaccard_index(set1: Iterable, set2: Iterable) -> float:
        A, B = set(set1), set(set2)
        union = A | B
        if not union:
            return 0.0
        return len(A & B) / len(union)
    
    def evaluate_community(self, C: List[Set], Cc: Set, t: float) -> List[Set]:
        Jm = 0.0  # Maximum Jaccard index found
        Cm: Set = set()  # Best merge candidate
        best_Cn: Set = set()  # Komunitas yang akan di-merge
        
        C_updated: List[Set] = []
        
        for Cn in C:
            Cn = set(Cn)
            
            # Line 28-29: if Cc ⊆ Cn then return C
            if Cc.issubset(Cn):
                # Kandidat sudah ter-cover oleh komunitas existing
                return C
            
            # Line 30-31: else if Cn ⊆ Cc then C = C \ {Cn}
            if Cn.issubset(Cc):
                # Cn lama akan digantikan oleh kandidat yang lebih besar
                # Jangan masukkan Cn ke C_updated
                continue
            
            # Line 32-35: else if Jc > Jm and Jc > t and αc ≥ t then
            Jc = self.jaccard_index(Cn, Cc)
            alpha_c = self.intraconnectivity(Cn | Cc)
            
            if Jc > Jm and Jc > t and alpha_c >= t:
                Jm = Jc  # Line 33
                Cm = Cn | Cc  # Line 34: merge (interpretasi: Cm bukan Cn)
                best_Cn = Cn
            
            # PERBAIKAN: Simpan SEMUA Cn yang tidak subset
            # (baik yang kandidat merge maupun tidak)
            # Nanti akan dihapus jika menjadi best_Cn
            C_updated.append(Cn)
        
        # Line 37-39: if Cm ≠ ∅ then Cc = Cm
        if Cm:
            # Hapus komunitas yang di-merge
            C_updated = [S for S in C_updated if S != best_Cn]
            Cc = Cm
        
        # Line 40: return C ∪ {Cc}
        C_updated.append(Cc)
        return C_updated
    
    def apal_algorithm(self, t=0.35):
        print(f"🚀 Running APAL algorithm with threshold t={t}")
        
        if not (0.0 <= t <= 1.0):
            raise ValueError("t must be in [0, 1]")
        
        C: List[Set] = []  # Community list (initially empty)
        
        # Reset dan inisialisasi tracking waktu
        self.execution_timeline = []
        start_time = time.time()
        
        # Iterasi deterministik: urutkan node untuk hasil konsisten
        V = sorted(list(self.graph.nodes()))
        total_vertices = len(V)
        processed = 0
        
        # Sampling interval untuk timeline (setiap 5% progress atau setiap 10 node)
        sample_interval = max(1, total_vertices // 20)
        
        # Line 2: for all v ∈ V do
        for v in V:
            processed += 1
            
            if processed % 50 == 0:
                print(f"   Processing: {processed}/{total_vertices} vertices...")
            
            # Line 3: for all vn ∈ N(v) do
            Nv = self.get_neighbors(v)  # N(v)
            
            for vn in sorted(Nv):  # Sort untuk konsistensi
                Nvn = self.get_neighbors(vn)  # N(vn)
                
                # Line 4: Cc ← N(v) ∩ N(vn)
                Cc = Nv.intersection(Nvn)
                
                # Line 5: if |Cc| > 0
                if not Cc:
                    continue
                
                # Line 6: Cc ← Cc ∪ {vn, v}
                Cc = Cc | {v, vn}
                
                # Line 7: if Intraconnectivity(Cc) ≥ t then
                if self.intraconnectivity(Cc) >= t:
                    # Line 8: C ← Evaluate(C, Cc, t)
                    C = self.evaluate_community(C, Cc, t)
            
            # Rekam waktu kumulatif pada interval tertentu
            if processed % sample_interval == 0 or processed == total_vertices:
                elapsed = time.time() - start_time
                progress_pct = (processed / total_vertices) * 100
                self.execution_timeline.append({
                    'node_index': processed,
                    'progress_percent': progress_pct,
                    'cumulative_time': elapsed,
                    'communities_count': len(C),
                    'current_node': v
                })
        
        # Rekam waktu sebelum deduplikasi
        dedup_start = time.time()
        
        # Line 13: Pembersihan duplikat & subset
        C = self._deduplicate(C)
        
        # Rekam waktu akhir (setelah deduplikasi)
        total_elapsed = time.time() - start_time
        dedup_time = time.time() - dedup_start
        
        # Tambahkan entry final
        self.execution_timeline.append({
            'node_index': total_vertices,
            'progress_percent': 100.0,
            'cumulative_time': total_elapsed,
            'communities_count': len(C),
            'current_node': 'FINAL',
            'deduplication_time': dedup_time
        })
        
        self.communities = C
        self.total_execution_time = total_elapsed
        
        print(f"✅ Detected {len(C)} communities in {total_elapsed:.4f} seconds")
        
        # Update node to communities mapping
        self._update_node_to_communities()
        
        return self.communities
    
    def get_execution_timeline(self):
        return self.execution_timeline
    
    @staticmethod
    def _deduplicate(C: List[Set]) -> List[Set]:
        # 1. Buang duplikat identik
        unique = []
        seen = set()
        for S in C:
            frozen = frozenset(S)
            if frozen not in seen:
                seen.add(frozen)
                unique.append(set(S))
        
        # 2. Buang subset
        keep = [True] * len(unique)
        for i in range(len(unique)):
            if not keep[i]:
                continue
            for j in range(len(unique)):
                if i == j or not keep[j]:
                    continue
                if unique[i].issubset(unique[j]):
                    # Jika identik, hanya satu yang dipertahankan
                    if len(unique[i]) < len(unique[j]) or i > j:
                        keep[i] = False
                        break
        
        result = [unique[i] for i in range(len(unique)) if keep[i]]
        
        # Sort untuk konsistensi output
        result.sort(key=lambda c: (len(c), min(c) if c else ""))
        
        return result
    
    def _update_node_to_communities(self):
        """Update mapping of nodes to their communities"""
        self.node_to_communities.clear()
        for i, community in enumerate(self.communities):
            for node in community:
                self.node_to_communities[node].add(i)

    def normalized_node_cut(self, community: Iterable) -> float:
        community = set(community)
        if len(community) <= 1:
            return 1.0  # Komunitas trivial dianggap bocor
        
        kin_total = 0.0  # k_in(C)
        boundary_conductance_sum = 0.0  # Σ [ (k_in_i * k_out_i) / k_i ]
        
        for node in community:
            neighbors = set(self.graph.neighbors(node))
            k_in_i = len(neighbors.intersection(community))  # Internal degree
            k_out_i = len(neighbors - community)  # External degree
            k_i = len(neighbors)  # Total degree
            
            kin_total += k_in_i
            
            if k_i > 0:
                boundary_conductance_sum += (k_in_i * k_out_i) / k_i
        
        if kin_total == 0:
            return 1.0
        
        return boundary_conductance_sum / kin_total

    def calculate_alternative_metrics(self, communities):
        metrics = {}
        
        # 1. Normalized Node Cut (Havemann et al. 2012) - PRIMARY METRIC (UNWEIGHTED)
        normalized_node_cuts = []
        for community in communities:
            if len(community) <= 1:
                continue
            
            psi = self.normalized_node_cut(community)  # UNWEIGHTED
            normalized_node_cuts.append(psi)
        
        metrics['avg_normalized_node_cut'] = np.mean(normalized_node_cuts) if normalized_node_cuts else 0.0
        metrics['normalized_node_cuts'] = normalized_node_cuts
        
        # 2. Overlapping Quality
        overlapping_quality = 0
        total_overlaps = 0
        
        node_memberships = defaultdict(list)
        for i, community in enumerate(communities):
            for node in community:
                node_memberships[node].append(i)
        
        for node, memberships in node_memberships.items():
            if len(memberships) > 1:
                total_overlaps += 1
                node_neighbors = set(self.graph.neighbors(node))
                
                community_neighbors = defaultdict(int)
                for neighbor in node_neighbors:
                    for membership in memberships:
                        if neighbor in communities[membership]:
                            community_neighbors[membership] += 1
                
                if len(community_neighbors) > 1:
                    overlapping_quality += 1
        
        metrics['overlap_quality'] = overlapping_quality / total_overlaps if total_overlaps > 0 else 0
        
        # 3. Coverage
        all_nodes_in_communities = set()
        for community in communities:
            all_nodes_in_communities.update(community)
        
        coverage = len(all_nodes_in_communities) / len(self.graph.nodes())
        metrics['coverage'] = coverage
        
        # 4. Biological Significance Score (based on size and conductance)
        bio_scores = []
        for i, community in enumerate(communities):
            size = len(community)
            # Use normalized node cut (lower is better, so invert for scoring)
            if i < len(normalized_node_cuts):
                conductance_score = max(0, 1 - normalized_node_cuts[i])  # Invert: lower cut = higher score
            else:
                conductance_score = 0
            
            # Size score (optimal range 3-50 proteins per complex)
            if 3 <= size <= 50:
                size_score = 1.0
            elif size < 3:
                size_score = size / 3.0
            else:
                size_score = max(0.1, 50.0 / size)
            
            bio_score = (size_score + conductance_score) / 2
            bio_scores.append(bio_score)
        
        metrics['avg_biological_score'] = np.mean(bio_scores) if bio_scores else 0
        
        return metrics
    
    def analyze_communities(self):
        """
        Comprehensive analysis using Normalized Node Cut as the primary quality metric.
        """
        if not hasattr(self, 'communities') or not self.communities:
            return {}
        
        # Calculate metrics with normalized node cut as primary
        alt_metrics = self.calculate_alternative_metrics(self.communities)
        
        # Find overlapping nodes
        overlapping_nodes = defaultdict(list)
        for i, community in enumerate(self.communities):
            for node in community:
                overlapping_nodes[node].append(i)
        
        overlapping_nodes = {node: comms for node, comms in overlapping_nodes.items() 
                           if len(comms) > 1}
        
        # Calculate intraconnectivity for each community (kept for compatibility)
        intraconn_values = []
        for community in self.communities:
            intraconn = self.intraconnectivity(community)
            intraconn_values.append(intraconn)
        
        avg_intraconnectivity = np.mean(intraconn_values) if intraconn_values else 0
        
        # Community sizes
        community_sizes = [len(c) for c in self.communities]
        
        analysis_results = {
            # PRIMARY METRICS (Normalized Node Cut - Havemann et al. 2012)
            'avg_normalized_node_cut': alt_metrics['avg_normalized_node_cut'],
            'normalized_node_cuts': alt_metrics['normalized_node_cuts'],
            
            # SECONDARY METRICS
            'overlap_quality': alt_metrics['overlap_quality'],
            'avg_biological_score': alt_metrics['avg_biological_score'],
            
            # BASIC STATS  
            'avg_intraconnectivity': avg_intraconnectivity,
            'intraconnectivity_values': intraconn_values,
            'overlapping_nodes': overlapping_nodes,
            'coverage': alt_metrics['coverage'],
            'community_sizes': community_sizes,
            'num_communities': len(self.communities),
            'num_overlapping_nodes': len(overlapping_nodes),
            'total_nodes': len(self.graph.nodes()),
            'total_edges': len(self.graph.edges()),
            'avg_community_size': np.mean(community_sizes) if community_sizes else 0
        }
        
        return analysis_results
    
    def get_community_stats(self):
        """Get detailed community statistics using normalized node cut as primary metric"""
        stats = []
        for i, community in enumerate(self.communities):
            community_list = list(community)
            intraconn = self.intraconnectivity(community)
            # Use normalized node cut as primary metric (UNWEIGHTED)
            normalized_cut = self.normalized_node_cut(community)
            
            stats.append({
                'community_id': i,
                'size': len(community),
                'normalized_node_cut': normalized_cut,  # PRIMARY
                'intraconnectivity': intraconn,  # SECONDARY
                'nodes': sorted(community_list),
                'meets_conductance_threshold': normalized_cut <= 0.3,  # Lower is better
                'meets_intraconnectivity_threshold': intraconn >= 0.35  # Original threshold check
            })
        return stats

    def visualize_overlapping_communities(self, max_communities=10, figsize=(20, 12)):
        """Visualize overlapping communities with enhanced color coding"""
        if not self.communities:
            print("No communities detected. Run APAL algorithm first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Get communities to visualize
        communities_to_show = self.communities[:max_communities]
        
        # Create subgraph with nodes from detected communities
        community_nodes = set()
        for community in communities_to_show:
            community_nodes.update(community)
        
        subgraph = self.graph.subgraph(community_nodes)
        
        # Create layout
        pos = nx.spring_layout(subgraph, k=2, iterations=100, seed=42)
        
        # Identify overlapping nodes
        node_membership_count = defaultdict(int)
        for community in communities_to_show:
            for node in community:
                node_membership_count[node] += 1
        
        overlapping_nodes = {n for n, count in node_membership_count.items() if count > 1}
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, edge_color='gray', width=0.5)
        
        # Color palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(communities_to_show)))
        
        # Draw non-overlapping nodes for each community
        for i, community in enumerate(communities_to_show):
            non_overlap_nodes = [n for n in community if n not in overlapping_nodes and n in subgraph.nodes()]
            if non_overlap_nodes:
                nx.draw_networkx_nodes(subgraph, pos, 
                                     nodelist=non_overlap_nodes,
                                     node_color=[colors[i]], 
                                     alpha=0.8,
                                     node_size=400,
                                     node_shape='o',
                                     edgecolors='black',
                                     linewidths=1.5,
                                     label=f'Community {i+1}')
        
        # Draw overlapping nodes with special markers
        if overlapping_nodes:
            overlap_list = [n for n in overlapping_nodes if n in subgraph.nodes()]
            nx.draw_networkx_nodes(subgraph, pos,
                                 nodelist=overlap_list,
                                 node_color='red',
                                 alpha=0.9,
                                 node_size=600,
                                 node_shape='*',
                                 edgecolors='darkred',
                                 linewidths=2,
                                 label='Overlapping Proteins')
        
        # Draw labels
        labels = {}
        for node in subgraph.nodes():
            if node in overlapping_nodes:
                labels[node] = f"{node}\n({node_membership_count[node]} communities)"
            elif len(labels) < 30:  # Limit labels for clarity
                labels[node] = node
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=7, 
                               font_weight='bold', font_color='darkblue')
        
        plt.title('Overlapping Community Detection using APAL Algorithm\n' + 
                 f'Total Communities: {len(communities_to_show)} | ' +
                 f'Overlapping Nodes: {len(overlapping_nodes)}',
                 fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('d:\\Intan APAL\\community_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: d:\\Intan APAL\\community_visualization.png")
    
    def visualize_community_heatmap(self, figsize=(14, 10)):
        """
        Create a heatmap showing community membership matrix
        """
        if not self.communities:
            print("No communities detected. Run APAL algorithm first.")
            return
        
        # Create membership matrix
        all_nodes = sorted(set().union(*self.communities))
        membership_matrix = np.zeros((len(all_nodes), len(self.communities)))
        
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        
        for comm_idx, community in enumerate(self.communities):
            for node in community:
                membership_matrix[node_to_idx[node], comm_idx] = 1
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(membership_matrix, 
                   cmap='YlOrRd', 
                   cbar_kws={'label': 'Membership'},
                   linewidths=0.5,
                   linecolor='gray')
        
        plt.xlabel('Community ID', fontsize=12, fontweight='bold')
        plt.ylabel('Protein Nodes', fontsize=12, fontweight='bold')
        plt.title('Community Membership Heatmap\n(Shows Overlapping Structure)', 
                 fontsize=14, fontweight='bold')
        
        # Show only subset of y-axis labels for clarity
        step = max(1, len(all_nodes) // 20)
        yticks_pos = range(0, len(all_nodes), step)
        yticks_labels = [all_nodes[i] for i in yticks_pos]
        plt.yticks(yticks_pos, yticks_labels, fontsize=8, rotation=0)
        
        plt.xticks(range(len(self.communities)), 
                  [f'C{i+1}' for i in range(len(self.communities))], 
                  fontsize=9, rotation=45)
        
        plt.tight_layout()
        plt.savefig('d:\\Intan APAL\\community_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Heatmap saved to: d:\\Intan APAL\\community_heatmap.png")
    
    def visualize_overlap_statistics(self, figsize=(16, 10)):
        """
        Create comprehensive overlap statistics visualization
        """
        if not self.communities:
            print("No communities detected. Run APAL algorithm first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Community Overlap Analysis', fontsize=16, fontweight='bold')
        
        # 1. Community size distribution
        sizes = [len(c) for c in self.communities]
        axes[0, 0].hist(sizes, bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Community Size')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Community Size Distribution')
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Node membership distribution
        node_counts = defaultdict(int)
        for community in self.communities:
            for node in community:
                node_counts[node] += 1
        
        membership_counts = list(node_counts.values())
        axes[0, 1].hist(membership_counts, bins=range(1, max(membership_counts)+2), 
                       color='lightcoral', edgecolor='black', align='left')
        axes[0, 1].set_xlabel('Number of Communities')
        axes[0, 1].set_ylabel('Number of Nodes')
        axes[0, 1].set_title('Node Membership Distribution')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Overlapping vs Non-overlapping
        overlapping_count = sum(1 for count in membership_counts if count > 1)
        non_overlapping_count = sum(1 for count in membership_counts if count == 1)
        
        axes[0, 2].bar(['Non-overlapping', 'Overlapping'], 
                      [non_overlapping_count, overlapping_count],
                      color=['lightgreen', 'orange'], edgecolor='black')
        axes[0, 2].set_ylabel('Number of Nodes')
        axes[0, 2].set_title('Node Classification')
        axes[0, 2].grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate([non_overlapping_count, overlapping_count]):
            axes[0, 2].text(i, v + max(membership_counts)*0.02, str(v), 
                           ha='center', fontweight='bold')
        
        # 4. Intraconnectivity distribution
        intraconn_values = [self.intraconnectivity(c) for c in self.communities]
        axes[1, 0].hist(intraconn_values, bins=20, color='mediumpurple', edgecolor='black')
        axes[1, 0].set_xlabel('Intraconnectivity (α)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Intraconnectivity Distribution')
        axes[1, 0].axvline(np.mean(intraconn_values), color='red', 
                          linestyle='--', label=f'Mean: {np.mean(intraconn_values):.3f}')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 5. Normalized Node Cut distribution
        if hasattr(self, 'communities'):
            ncut_values = [self.normalized_node_cut(c) for c in self.communities]
            axes[1, 1].hist(ncut_values, bins=20, color='gold', edgecolor='black')
            axes[1, 1].set_xlabel('Normalized Node Cut (Ψ)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Quality Metric Distribution')
            axes[1, 1].axvline(np.mean(ncut_values), color='red', 
                              linestyle='--', label=f'Mean: {np.mean(ncut_values):.3f}')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        
        # 6. Top overlapping nodes
        top_overlapping = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_overlapping:
            nodes, counts = zip(*top_overlapping)
            axes[1, 2].barh(range(len(nodes)), counts, color='teal', edgecolor='black')
            axes[1, 2].set_yticks(range(len(nodes)))
            axes[1, 2].set_yticklabels(nodes, fontsize=8)
            axes[1, 2].set_xlabel('Number of Communities')
            axes[1, 2].set_title('Top 10 Overlap Proteins')
            axes[1, 2].grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('d:\\Intan APAL\\overlap_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Statistics visualization saved to: d:\\Intan APAL\\overlap_statistics.png")

# Keep the original class for backward compatibility
class APALCommunityDetector:
    """
    Adjacency Propagation Algorithm for overlapping community detection
    Wrapper class untuk backward compatibility
    """
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.graph = None
        self.communities = []
        self.node_to_communities = defaultdict(set)
        
    def load_data(self, filepath):
        """Load protein interaction data from TSV file"""
        print("Loading protein interaction data...")
        
        df = pd.read_csv(filepath, sep='\t')
        df.columns = df.columns.str.replace('#', '', regex=False)
        
        self.graph = nx.Graph()
        
        for _, row in df.iterrows():
            node1 = row['node1']
            node2 = row['node2']
            self.graph.add_edge(node1, node2)
        
        print(f"Graph created with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def apal_algorithm(self, t=0.5):
        """Run APAL algorithm using corrected implementation"""
        print(f"Running APAL algorithm with threshold t={t}...")
        
        apal_detector = APALDetector(self.graph)
        communities = apal_detector.apal_algorithm(t=t)
        
        self.communities = communities
        self._update_node_to_communities()
        
        return communities
    
    def _update_node_to_communities(self):
        self.node_to_communities.clear()
        for i, community in enumerate(self.communities):
            for node in community:
                self.node_to_communities[node].add(i)
    
    def get_overlapping_nodes(self):
        """
        Get nodes that belong to multiple communities
        
        Returns:
            dict: Dictionary mapping nodes to their communities
        """
        overlapping = {}
        for node, communities in self.node_to_communities.items():
            if len(communities) > 1:
                overlapping[node] = communities
        return overlapping
    
    def analyze_communities(self):
        """
        Analyze the detected communities
        
        Returns:
            dict: Analysis results
        """
        if not self.communities:
            print("No communities detected. Run APAL algorithm first.")
            return {}
        
        analysis = {
            'num_communities': len(self.communities),
            'community_sizes': [len(c) for c in self.communities],
            'overlapping_nodes': self.get_overlapping_nodes(),
            'coverage': self._calculate_coverage()
        }
        
        print("=== COMMUNITY ANALYSIS ===")
        print(f"Number of communities: {analysis['num_communities']}")
        print(f"Community sizes: {analysis['community_sizes']}")
        print(f"Average community size: {np.mean(analysis['community_sizes']):.2f}")
        print(f"Number of overlapping nodes: {len(analysis['overlapping_nodes'])}")
        print(f"Coverage: {analysis['coverage']:.4f}")
        
        return analysis
    
    def _calculate_coverage(self):
        covered_nodes = set()
        for community in self.communities:
            covered_nodes.update(community)
        
        return len(covered_nodes) / len(self.graph.nodes()) if len(self.graph.nodes()) > 0 else 0.0
    
    def visualize_communities(self, max_communities=10, figsize=(15, 10)):
        if not self.communities:
            print("No communities detected. Run APAL algorithm first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Create subgraph with nodes from detected communities
        community_nodes = set()
        for community in self.communities[:max_communities]:
            community_nodes.update(community)
        
        subgraph = self.graph.subgraph(community_nodes)
        
        # Create layout
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Color nodes by community (overlapping nodes will show multiple colors)
        colors = plt.cm.Set3(np.linspace(0, 1, min(len(self.communities), max_communities)))
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, edge_color='gray')
        
        # Draw nodes for each community
        for i, community in enumerate(self.communities[:max_communities]):
            community_nodes_in_subgraph = [n for n in community if n in subgraph.nodes()]
            if community_nodes_in_subgraph:
                nx.draw_networkx_nodes(subgraph, pos, 
                                     nodelist=community_nodes_in_subgraph,
                                     node_color=[colors[i]], 
                                     alpha=0.7,
                                     node_size=300,
                                     label=f'Community {i+1}')
        
        # Draw labels for important nodes
        important_nodes = list(community_nodes)[:20]  # Show labels for first 20 nodes
        labels = {node: node for node in important_nodes if node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        plt.title('Protein Interaction Network - Community Detection using APAL')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_file='apal_results.txt'):
        """
        Save community detection results to file
        
        Args:
            output_file (str): Output file path
        """
        filepath = f"d:\\Intan APAL\\{output_file}"
        
        with open(filepath, 'w') as f:
            f.write("APAL Community Detection Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Number of communities: {len(self.communities)}\n")
            f.write(f"Threshold used: {self.threshold}\n\n")
            
            for i, community in enumerate(self.communities):
                f.write(f"Community {i+1} (size: {len(community)}):\n")
                f.write(f"Nodes: {', '.join(sorted(community))}\n")
                f.write(f"Intraconnectivity: {self.intraconnectivity(community):.4f}\n\n")
            
            # Overlapping nodes
            overlapping = self.get_overlapping_nodes()
            if overlapping:
                f.write("Overlapping Nodes:\n")
                for node, communities in overlapping.items():
                    f.write(f"{node}: Communities {list(communities)}\n")
        
        print(f"Results saved to {filepath}")

    def visualize_individual_communities(self, output_dir='d:\\Intan APAL\\community_visualizations'):
        """
        Visualize each community individually with professional layout
        Similar to the provided example visualization
        
        Args:
            output_dir (str): Directory to save individual community visualizations
        """
        if not self.communities:
            print("No communities detected. Run APAL algorithm first.")
            return
        
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n=== Generating Individual Community Visualizations ===")
        
        for idx, community in enumerate(self.communities):
            community_id = idx + 1
            print(f"  Visualizing Community {community_id} ({len(community)} nodes)...")
            
            # Create subgraph for this community
            subgraph = self.graph.subgraph(community)
            
            if subgraph.number_of_nodes() == 0:
                continue
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Calculate layout using spring layout for better distribution
            pos = nx.spring_layout(subgraph, k=1.5, iterations=100, seed=42)
            
            # Identify overlapping nodes in this community
            overlapping_in_community = set()
            for node in community:
                if node in self.node_to_communities and len(self.node_to_communities[node]) > 1:
                    overlapping_in_community.add(node)
            
            # Separate nodes into overlapping and non-overlapping
            non_overlap_nodes = [n for n in community if n not in overlapping_in_community]
            overlap_nodes = list(overlapping_in_community)
            
            # Draw edges first (in background)
            nx.draw_networkx_edges(subgraph, pos, 
                                  alpha=0.3, 
                                  edge_color='gray', 
                                  width=1.0)
            
            # Draw non-overlapping nodes (regular community members)
            if non_overlap_nodes:
                nx.draw_networkx_nodes(subgraph, pos,
                                     nodelist=non_overlap_nodes,
                                     node_color='#1f77b4',  # Blue color
                                     node_size=300,
                                     alpha=0.8,
                                     edgecolors='black',
                                     linewidths=1.5)
            
            # Draw overlapping nodes with different color
            if overlap_nodes:
                nx.draw_networkx_nodes(subgraph, pos,
                                     nodelist=overlap_nodes,
                                     node_color='#ff7f0e',  # Orange color for overlap
                                     node_size=400,
                                     alpha=0.9,
                                     edgecolors='darkred',
                                     linewidths=2.0)
            
            # Draw labels for all nodes
            labels = {node: node for node in subgraph.nodes()}
            nx.draw_networkx_labels(subgraph, pos, labels,
                                   font_size=8,
                                   font_weight='bold',
                                   font_color='black')
            
            # Calculate modularity (simplified version)
            internal_edges = subgraph.number_of_edges()
            total_degree = sum(dict(subgraph.degree()).values())
            modularity = internal_edges / max(1, total_degree) if total_degree > 0 else 0
            
            # Title with statistics
            title = f'Struktur Komunitas (Modularity: {modularity:.4f}, Jumlah Komunitas: {len(self.communities)})\n'
            title += f'Community {community_id}: {len(community)} proteins'
            if overlap_nodes:
                title += f' ({len(overlap_nodes)} overlap proteins)'
            
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
            plt.axis('off')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#1f77b4', edgecolor='black', label='Regular Protein'),
            ]
            if overlap_nodes:
                legend_elements.append(
                    Patch(facecolor='#ff7f0e', edgecolor='darkred', label='Overlap Protein (Multi-community)')
                )
            plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            plt.tight_layout()
            
            # Save figure
            filename = f'community_{community_id:03d}_size{len(community)}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"    ✓ Saved: {filename}")
        
        print(f"\n✓ All {len(self.communities)} community visualizations saved to: {output_dir}")
        
        # Create summary visualization showing all communities
        self._create_all_communities_summary(output_dir)
    
    def _create_all_communities_summary(self, output_dir):
        """Create a summary grid showing all communities"""
        import math
        
        num_communities = len(self.communities)
        
        # Determine grid size (try to make it square-ish)
        cols = math.ceil(math.sqrt(num_communities))
        rows = math.ceil(num_communities / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle(f'All {num_communities} Communities - Overview', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Flatten axes array for easy iteration
        if num_communities == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, (ax, community) in enumerate(zip(axes, self.communities)):
            subgraph = self.graph.subgraph(community)
            
            if subgraph.number_of_nodes() == 0:
                ax.axis('off')
                continue
            
            pos = nx.spring_layout(subgraph, k=1.5, iterations=50, seed=42)
            
            # Identify overlapping nodes
            overlapping = [n for n in community 
                          if n in self.node_to_communities and len(self.node_to_communities[n]) > 1]
            non_overlapping = [n for n in community if n not in overlapping]
            
            # Draw
            nx.draw_networkx_edges(subgraph, pos, alpha=0.2, edge_color='gray', width=0.5, ax=ax)
            
            if non_overlapping:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=non_overlapping,
                                     node_color='#1f77b4', node_size=100, 
                                     alpha=0.7, ax=ax)
            if overlapping:
                nx.draw_networkx_nodes(subgraph, pos, nodelist=overlapping,
                                     node_color='#ff7f0e', node_size=150, 
                                     alpha=0.9, ax=ax)
            
            ax.set_title(f'C{idx+1} (n={len(community)})', fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_communities, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, '00_ALL_COMMUNITIES_SUMMARY.png')
        plt.savefig(summary_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    ✓ Summary visualization saved: 00_ALL_COMMUNITIES_SUMMARY.png")

def main():
    """Main function to run the APAL community detection"""
    print("=== DETEKSI KOMUNITAS OVERLAP MENGGUNAKAN APAL ===")
    print("=== IMPLEMENTASI BENAR dengan Evaluate Function yang Diperbaiki ===\n")
    
    data_file = "d:\\Intan APAL\\string_interactions.tsv"
    try:
        detector = APALCommunityDetector(threshold=0.7)
        graph = detector.load_data(data_file)
        
        print(f"✅ Grafik berhasil dimuat dengan {graph.number_of_nodes()} node dan {graph.number_of_edges()} edge")
        
        apal = APALDetector(graph)
        
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        best_communities = []
        best_threshold = 0.3
        best_score = float('inf')
        
        print("Menguji threshold yang berbeda...")
        for t in thresholds:
            print(f"\n--- Threshold APAL: {t} ---")
            communities = apal.apal_algorithm(t=t)
            analysis = apal.analyze_communities()
            
            print(f"Komunitas ditemukan: {analysis['num_communities']}")
            print(f"Normalized Node Cut (Ψ): {analysis['avg_normalized_node_cut']:.4f}")
            
            if analysis.get('avg_normalized_node_cut', float('inf')) < best_score:
                best_score = analysis['avg_normalized_node_cut']
                best_threshold = t
                best_communities = communities.copy()
        
        print(f"\n=== HASIL TERBAIK (Threshold APAL: {best_threshold}) ===")
        
        apal.communities = best_communities
        apal._update_node_to_communities()
        final_analysis = apal.analyze_communities()
        
        print(f"\n=== HASIL ===")
        print(f"  • Komunitas terdeteksi: {final_analysis['num_communities']}")
        print(f"  • Normalized Node Cut (Ψ): {final_analysis['avg_normalized_node_cut']:.4f}")
        print(f"  • Coverage: {final_analysis['coverage']:.4f}")
        print(f"  • Overlap protein: {final_analysis['num_overlapping_nodes']}")
        
        detector.graph = graph
        detector.communities = best_communities
        detector._update_node_to_communities()
        detector.save_results('hasil_apal_corrected.txt')
        
        print(f"\n✅ Algoritma APAL dengan Evaluate function yang benar telah dijalankan!")
        
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan: {data_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()