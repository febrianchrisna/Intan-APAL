import dash
from dash import dcc, html, Input, Output, State, dash_table, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import base64
import io
import os
import time

# Import the corrected APAL implementation
from apal_breast_cancer import APALDetector, APALCommunityDetector

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "APAL Community Detection Interface"

# Enable callback exceptions since we're creating components dynamically
app.config.suppress_callback_exceptions = True

# Global variables
G = None
detector = None
current_results = None
execution_time_history = []  # Store execution time history for visualization

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("APAL Community Detection System", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        html.P("Algoritma Adjacency Propagation untuk Deteksi Komunitas Overlap pada Jaringan Biologis",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px'})
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'marginBottom': '20px'}),
    
    # Data Input Section
    html.Div([
        html.H3("📁 Input Data", style={'color': '#34495e'}),
        html.Div([
            html.Div([
                html.Label("Upload File Interaksi Protein:", style={'fontWeight': 'bold'}),
                html.P("Kolom yang diperlukan: node1, node2", style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '5px 0'}),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        '📁 Klik untuk memilih file atau drag & drop file di sini'
                    ]),
                    style={
                        'width': '100%',
                        'height': '80px',
                        'lineHeight': '80px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '10px',
                        'borderColor': '#3498db',
                        'textAlign': 'center',
                        'margin': '10px 0',
                        'backgroundColor': '#f8f9fa',
                        'cursor': 'pointer'
                    },
                    multiple=False
                )
            ], style={'width': '100%', 'display': 'inline-block'})
        ]),
        
        html.Div(id='data-status', style={'marginTop': '15px'})
    ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # Network Analysis Section
    html.Div([
        html.H3("🔬 Analisis Jaringan", style={'color': '#34495e'}),
        html.Div(id='network-stats', style={'marginBottom': '20px'}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='network-viz')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='degree-dist')
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # APAL Parameters Section
    html.Div([
        html.H3("⚙️ Parameter Algoritma APAL", style={'color': '#34495e'}),
        html.Div([
            html.Div([
                html.Label("Threshold APAL (t):", style={'fontWeight': 'bold', 'color': '#e74c3c'}),
                html.P("Intraconnectivity minimum untuk komunitas yang valid", 
                       style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '5px 0'}),
                html.P("Rekomendasi: 0.1-0.4 untuk jaringan protein", 
                       style={'fontSize': '12px', 'color': '#27ae60', 'margin': '5px 0', 'fontWeight': 'bold'}),
                dcc.Slider(
                    id='apal-threshold',
                    min=0.1,
                    max=1.0,
                    step=0.1,
                    value=0.3,
                    marks={i/10: f'{i/10}' for i in range(1, 11)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '70%', 'display': 'inline-block'}),
            
            html.Div([
                html.Button('Jalankan APAL', id='run-apal-btn',
                           style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                                 'padding': '15px 30px', 'borderRadius': '5px', 'fontSize': '16px'})
            ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center', 'verticalAlign': 'middle'})
        ])
    ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # Results Section
    html.Div(id='results-section', children=[
        html.H3("📊 Hasil APAL", style={'color': '#34495e'}),
        html.Div("Jalankan APAL untuk melihat hasil...", style={'textAlign': 'center', 'color': '#7f8c8c', 'fontSize': '16px'})
    ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # Execution Time Section (NEW)
    html.Div(id='execution-time-section', children=[
        html.H3("⏱️ Pengukuran Waktu Eksekusi", style={'color': '#34495e'}),
        html.Div("Jalankan APAL untuk melihat waktu eksekusi...", style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
    ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # Community Visualization Section (NEW)
    html.Div(id='community-viz-section', style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # Community Details Section
    html.Div(id='community-details-section', style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '10px'}),
    
    # Export button (moved outside of dynamic content)
    html.Div([
        html.Button(
            "📊 Ekspor Hasil (CSV)",
            id="export-csv-btn",
            style={
                'backgroundColor': '#28a745',
                'color': 'white',
                'border': 'none',
                'padding': '12px 25px',
                'borderRadius': '8px',
                'fontSize': '16px',
                'fontWeight': 'bold',
                'cursor': 'pointer',
                'display': 'none'  # Initially hidden
            }
        )
    ], id='export-button-container', style={'textAlign': 'center', 'padding': '20px'}),
    
    # Download component for CSV export
    dcc.Download(id="download-communities-csv"),
    # Add a store to track export button state
    dcc.Store(id="export-store", data={"clicks": 0})
])

# Callback for loading data - MANUAL UPLOAD ONLY
@app.callback(
    Output('data-status', 'children'),
    Output('network-stats', 'children'),
    Output('network-viz', 'figure'),
    Output('degree-dist', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_data(upload_contents, upload_filename):
    global G, detector, current_results
    
    if upload_contents is None:
        return "Silakan upload file interaksi protein", "", {}, {}
    
    try:
        # Handle uploaded file
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Parse uploaded file
        if 'csv' in upload_filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'tsv' in upload_filename.lower() or 'txt' in upload_filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t')
        else:
            return "❌ Format file tidak didukung. Gunakan file CSV, TSV, atau TXT.", "", {}, {}
        
        # Clean column names
        df.columns = df.columns.str.replace('#', '', regex=False)
        
        # Check required columns
        required_cols = ['node1', 'node2']
        if not all(col in df.columns for col in required_cols):
            return f"❌ Kolom yang diperlukan tidak ditemukan: {required_cols}", "", {}, {}
        
        # Create graph - unweighted for normalized node cut calculations
        G = nx.Graph()
        for _, row in df.iterrows():
            node1, node2 = row['node1'], row['node2']
            G.add_edge(node1, node2)
        
        status_msg = f"✅ File berhasil diupload: {upload_filename}"
        
        # Create detector
        detector = APALDetector(G)
        current_results = None
        
        # Generate statistics
        stats = create_network_stats(G)
        viz_fig = create_network_visualization(G)
        degree_fig = create_degree_distribution(G)
        
        return status_msg, stats, viz_fig, degree_fig
        
    except Exception as e:
        return f"❌ Gagal memuat data: {str(e)}", "", {}, {}

# Callback for running APAL
@app.callback(
    Output('results-section', 'children'),
    Output('community-details-section', 'children'),
    Output('community-viz-section', 'children'),
    Output('execution-time-section', 'children'),
    Output('export-csv-btn', 'style'),
    Input('run-apal-btn', 'n_clicks'),
    State('apal-threshold', 'value')
)
def run_apal_analysis(n_clicks, apal_threshold):
    global G, detector, current_results, execution_time_history
    
    if n_clicks is None or G is None or detector is None:
        return [
            html.H3("📊 Hasil APAL", style={'color': '#34495e'}),
            html.Div("Silakan upload data terlebih dahulu, kemudian jalankan APAL...", style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
        ], "", "", [
            html.H3("⏱️ Pengukuran Waktu Eksekusi", style={'color': '#34495e'}),
            html.Div("Jalankan APAL untuk melihat waktu eksekusi...", style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
        ], {'display': 'none'}
    
    try:
        # Clear detector state and reinitialize for fresh run
        detector = APALDetector(G)
        
        # ===== TIMING: Start measuring execution time =====
        start_time = time.time()
        
        # Run APAL algorithm
        print(f"Starting APAL analysis with threshold {apal_threshold}...")
        communities = detector.apal_algorithm(t=apal_threshold)
        
        # ===== TIMING: End measuring APAL algorithm time =====
        apal_time = time.time() - start_time
        
        # Continue with analysis (not timed - only APAL execution matters)
        analysis = detector.analyze_communities()
        
        # Store timing data (only APAL time)
        timing_data = {
            'run_number': len(execution_time_history) + 1,
            'threshold': apal_threshold,
            'apal_time': apal_time,
            'num_communities': len(communities),
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
        }
        execution_time_history.append(timing_data)
        
        # Ambil timeline eksekusi kumulatif dari detector
        execution_timeline = detector.get_execution_timeline() if hasattr(detector, 'get_execution_timeline') else []
        
        current_results = {
            'communities': communities, 
            'analysis': analysis, 
            'threshold': apal_threshold, 
            'timing': timing_data,
            'execution_timeline': execution_timeline
        }
        
        print(f"APAL completed. Found {len(communities)} communities.")
        print(f"Average Internal Density (ρ): {analysis.get('avg_internal_density', 0):.4f}")
        
        # Calculate overlap statistics
        overlapping_nodes = analysis.get('overlapping_nodes', {})
        overlapping_communities = 0
        non_overlapping_communities = 0
        
        # Check which communities have overlapping nodes
        for i, community in enumerate(communities):
            has_overlap = any(node in overlapping_nodes for node in community)
            if has_overlap:
                overlapping_communities += 1
            else:
                non_overlapping_communities += 1
        
        # Create results summary with CONSISTENT modern cards
        def metric_card(value, label, sublabel, color, border_color):
            return html.Div([
                html.Div(str(value), style={
                    'fontSize': '42px', 'fontWeight': '700', 'color': color, 'lineHeight': '1', 'marginBottom': '8px'
                }),
                html.Div(label, style={
                    'fontSize': '14px', 'fontWeight': '600', 'color': '#1e293b', 'marginBottom': '4px'
                }),
                html.Div(sublabel, style={
                    'fontSize': '11px', 'color': '#64748b'
                })
            ], style={
                'textAlign': 'center',
                'padding': '24px 20px',
                'background': 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
                'borderRadius': '16px',
                'flex': '1',
                'minWidth': '200px',
                'borderLeft': f'4px solid {border_color}',
                'boxShadow': '0 1px 3px rgba(0,0,0,0.08)'
            })
        
        results_summary = html.Div([
            html.H3("📊 Hasil Deteksi Komunitas APAL", style={
                'color': '#1e293b', 'marginBottom': '24px', 'fontSize': '20px', 'fontWeight': '700'
            }),
            
            # Main metrics row - CONSISTENT CARDS
            html.Div([
                metric_card(
                    len(communities), 
                    "Total Komunitas", 
                    f"{overlapping_communities} overlap · {non_overlapping_communities} non-overlap",
                    '#6366f1', '#6366f1'
                ),
                metric_card(
                    len(overlapping_nodes),
                    "Overlap Protein", 
                    "Protein di beberapa komunitas",
                    '#8b5cf6', '#8b5cf6'
                )
            ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '32px', 'flexWrap': 'wrap'}),
            
            # Charts row
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='community-sizes',
                        figure=create_community_size_chart(communities),
                        config={'displayModeBar': False}
                    )
                ], style={'width': '100%'})
            ])
        ])
        
        # Community details table
        community_details = create_community_details_table(communities, detector, overlapping_nodes, apal_threshold)
        
        # Community visualization section
        community_viz = create_community_visualization_section(communities)
        
        # Execution time visualization section
        execution_time_viz = create_execution_time_section(timing_data, execution_timeline)
        
        # Show export button
        export_btn_style = {
            'backgroundColor': '#28a745',
            'color': 'white',
            'border': 'none',
            'padding': '12px 25px',
            'borderRadius': '8px',
            'fontSize': '16px',
            'fontWeight': 'bold',
            'cursor': 'pointer',
            'display': 'inline-block'
        }
        
        return results_summary, community_details, community_viz, execution_time_viz, export_btn_style
        
    except Exception as e:
        error_msg = html.Div([
            html.H3("📊 Hasil APAL", style={'color': '#34495e'}),
            html.Div([
                html.Span("❌ ", style={'color': 'red', 'fontSize': '20px'}),
                html.Span(f"Error menjalankan APAL: {str(e)}")
            ], style={'color': 'red', 'textAlign': 'center'})
        ])
        return error_msg, "", "", [
            html.H3("⏱️ Pengukuran Waktu Eksekusi", style={'color': '#34495e'}),
            html.Div(f"Error: {str(e)}", style={'color': 'red', 'textAlign': 'center'})
        ], {'display': 'none'}

def create_execution_time_section(current_timing, execution_timeline=None):
    """Create MODERN execution time visualization section - CONSISTENT with other sections"""
    
    # Format time display - always 3 decimal places in seconds
    exec_time = current_timing['apal_time']
    time_display = f"{exec_time:.3f}s"
    
    # Consistent metric card function
    def metric_card(value, label, sublabel, color, border_color):
        return html.Div([
            html.Div(str(value), style={
                'fontSize': '42px', 'fontWeight': '700', 'color': color, 'lineHeight': '1', 'marginBottom': '8px'
            }),
            html.Div(label, style={
                'fontSize': '14px', 'fontWeight': '600', 'color': '#1e293b', 'marginBottom': '4px'
            }),
            html.Div(sublabel, style={
                'fontSize': '11px', 'color': '#64748b'
            })
        ], style={
            'textAlign': 'center',
            'padding': '24px 20px',
            'background': 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
            'borderRadius': '16px',
            'flex': '1',
            'minWidth': '140px',
            'borderLeft': f'4px solid {border_color}',
            'boxShadow': '0 1px 3px rgba(0,0,0,0.08)'
        })
    
    current_time_display = html.Div([
        # TITLE - Consistent with other sections
        html.H3("⏱️ Pengukuran Waktu Eksekusi Algoritma APAL", style={
            'color': '#1e293b', 'marginBottom': '24px', 'fontSize': '20px', 'fontWeight': '700'
        }),
        
        # Metrics row - CONSISTENT with results section
        html.Div([
            metric_card(time_display, "Waktu Eksekusi", "Durasi algoritma APAL", '#6366f1', '#6366f1'),
            metric_card(f"{current_timing['num_nodes']:,}", "Nodes", "Jumlah protein", '#8b5cf6', '#8b5cf6'),
            metric_card(f"{current_timing['num_edges']:,}", "Edges", "Jumlah interaksi", '#06b6d4', '#06b6d4'),
            metric_card(f"{current_timing['num_communities']}", "Komunitas", "Hasil deteksi", '#10b981', '#10b981'),
        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '32px', 'flexWrap': 'wrap'}),

        # === GRAFIK: Perkembangan Waktu Eksekusi Kumulatif ===
        html.Div([
            html.H4("📈 Perkembangan Waktu Eksekusi Kumulatif", style={
                'color': '#1e293b', 'marginBottom': '16px', 'fontSize': '16px', 'fontWeight': '600'
            }),
            dcc.Graph(
                id='cumulative-execution-time',
                figure=create_cumulative_time_chart(execution_timeline, current_timing),
                config={'displayModeBar': False}
            )
        ])
    ])
    
    return current_time_display


def create_cumulative_time_chart(execution_timeline, current_timing):
    """Create MODERN cumulative time chart - consistent with other charts"""
    fig = go.Figure()
    
    PRIMARY = '#6366f1'
    SECONDARY = '#10b981'
    
    if not execution_timeline:
        fig.add_annotation(
            text="Data timeline tidak tersedia",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=13, color='#94a3b8')
        )
        fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig
    
    # Ekstrak data
    progress_pct = [0] + [d['progress_percent'] for d in execution_timeline]
    cumulative_times = [0] + [d['cumulative_time'] for d in execution_timeline]
    communities_count = [0] + [d['communities_count'] for d in execution_timeline]
    
    # Waktu Kumulatif - Area chart
    fig.add_trace(go.Scatter(
        x=progress_pct,
        y=cumulative_times,
        mode='lines',
        name='Waktu',
        line=dict(color=PRIMARY, width=3, shape='spline'),
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.15)',
        hovertemplate='Progress: %{x:.0f}%<br>Waktu: %{y:.4f}s<extra></extra>'
    ))
    
    # Komunitas line
    fig.add_trace(go.Scatter(
        x=progress_pct,
        y=communities_count,
        mode='lines',
        name='Komunitas',
        line=dict(color=SECONDARY, width=2, shape='spline', dash='dot'),
        yaxis='y2',
        hovertemplate='Progress: %{x:.0f}%<br>Komunitas: %{y}<extra></extra>'
    ))
    
    # End marker
    if cumulative_times:
        fig.add_trace(go.Scatter(
            x=[progress_pct[-1]],
            y=[cumulative_times[-1]],
            mode='markers',
            marker=dict(size=12, color=PRIMARY, line=dict(width=2, color='white')),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Total time annotation
        fig.add_annotation(
            x=progress_pct[-1], y=cumulative_times[-1],
            text=f"<b>{current_timing['apal_time']:.3f}s</b>",
            showarrow=True, arrowhead=0, ax=-40, ay=-25,
            font=dict(size=11, color=PRIMARY),
            bgcolor='white', bordercolor=PRIMARY, borderwidth=1
        )
    
    fig.update_layout(
        xaxis=dict(
            title=dict(text='Progress (%)', font=dict(size=11, color='#94a3b8')),
            ticksuffix='%',
            range=[-2, 105],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=False,
            tickfont=dict(size=10, color='#94a3b8')
        ),
        yaxis=dict(
            title=dict(text='Waktu (s)', font=dict(size=11, color=PRIMARY)),
            tickfont=dict(color=PRIMARY, size=10),
            showgrid=True,
            gridcolor='rgba(99, 102, 241, 0.08)',
            zeroline=False
        ),
        yaxis2=dict(
            title=dict(text='Komunitas', font=dict(size=11, color=SECONDARY)),
            tickfont=dict(color=SECONDARY, size=10),
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=10)
        ),
        height=320,
        margin=dict(l=50, r=50, t=30, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        hoverlabel=dict(bgcolor='white', font_size=11, font_color='black', bordercolor='#e2e8f0')
    )
    
    return fig


def create_community_visualization_section(communities):
    """Create the visualization section with dropdown selector"""
    if not communities:
        return html.Div()
    
    # Create options for dropdown
    options = [{'label': '🌐 Semua Komunitas', 'value': -1}]
    options.extend([
        {'label': f'Komunitas {i+1} ({len(comm)} protein)', 'value': i} 
        for i, comm in enumerate(communities)
    ])
    
    return html.Div([
        html.H3("🎨 Visualisasi Komunitas", style={'color': '#34495e', 'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Pilih komunitas untuk divisualisasi:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='community-selector',
                options=options,
                value=-1,
                clearable=False,
                style={'width': '100%', 'marginBottom': '20px'}
            )
        ], style={'width': '40%', 'margin': '0 auto'}),
        
        dcc.Graph(id='individual-community-viz', style={'height': '700px'})
    ])

# Callback for individual community visualization
@app.callback(
    Output('individual-community-viz', 'figure'),
    Input('community-selector', 'value'),
    prevent_initial_call=True
)
def update_community_visualization(community_idx):
    global current_results, G, detector
    
    if current_results is None or community_idx is None:
        return go.Figure()
    
    communities = current_results['communities']
    overlapping_nodes = current_results['analysis'].get('overlapping_nodes', {})
    
    # If -1 is selected, show all communities overview
    if community_idx == -1:
        return create_all_communities_overview(communities, G, overlapping_nodes)
    
    # Show individual community
    if community_idx >= len(communities):
        return go.Figure()
    
    community = communities[community_idx]
    
    # Create subgraph
    subgraph = G.subgraph(community)
    
    if subgraph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="Komunitas kosong", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Get community color
    colors = px.colors.qualitative.Set3[:max(12, len(communities))]
    community_color = colors[community_idx % len(colors)]
    
    # Layout with better spacing
    pos = nx.spring_layout(subgraph, k=2.0, iterations=100, seed=42)
    
    # Identify overlapping nodes in this community
    overlap_in_comm = [n for n in community if n in overlapping_nodes]
    non_overlap_in_comm = [n for n in community if n not in overlapping_nodes]
    
    # Calculate normalized node cut for this community
    norm_cut = detector.normalized_node_cut(community)
    intraconn = detector.intraconnectivity(community)
    
    # Edges
    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='rgba(125, 125, 125, 0.3)'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add non-overlapping nodes
    if non_overlap_in_comm:
        node_x = [pos[node][0] for node in non_overlap_in_comm]
        node_y = [pos[node][1] for node in non_overlap_in_comm]
        node_text = [f"{node}<br>Komunitas {community_idx + 1}<br>Degree: {subgraph.degree(node)}" 
                    for node in non_overlap_in_comm]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color=community_color,
                line=dict(width=2, color='black')
            ),
            text=non_overlap_in_comm,
            textposition="top center",
            textfont=dict(size=10, color='black'),
            hovertext=node_text,
            hoverinfo='text',
            name=f'Komunitas {community_idx + 1}',
            showlegend=True
        ))
    
    # Add overlapping nodes with multi-color outlines
    if overlap_in_comm:
        for node in overlap_in_comm:
            x, y = pos[node]
            
            # Find all communities this node belongs to
            node_communities = [i for i, comm in enumerate(communities) if node in comm]
            comm_indices = [i+1 for i in node_communities]
            comm_colors = [colors[i % len(colors)] for i in node_communities]
            hover_text = f"{node}<br>Overlap Protein<br>Komunitas: {comm_indices}<br>Degree: {subgraph.degree(node)}"
            
            # Create concentric circles for multiple community membership
            base_size = 20
            
            # Draw outer circles first (larger to smaller) for each community
            for i in range(len(node_communities)-1, -1, -1):
                circle_size = base_size + (i * 5)  # Each outer circle is 5 pixels larger
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    marker=dict(
                        size=circle_size,
                        color='rgba(255,255,255,0)',  # Transparent fill
                        line=dict(width=3, color=comm_colors[i])
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add the center filled circle with the current community color
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(
                    size=base_size-5,  # Slightly smaller to show all outlines
                    color=community_color,  # Use the color of the current community being viewed
                    line=dict(width=1, color='black')
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add text label
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='text',
                text=[node],
                textposition="top center",
                textfont=dict(size=10, color='black', family='Arial Black'),
                hovertext=[hover_text],
                hoverinfo='text',
                showlegend=False
            ))
        
        # Add legend entry for overlap
        if overlap_in_comm:
            # Create sample multi-outline for legend
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=20, color='white', line=dict(width=3, color='black')),
                name='Overlap Protein (Multi-outline)',
                showlegend=True
            ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>Struktur Komunitas {community_idx + 1}</b><br>' +
                 f'<span style="font-size:14px">Ukuran: {len(community)} protein ({len(overlap_in_comm)} overlap, {len(non_overlap_in_comm)} non-overlap)<br>' +
                 f'Normalized Cut (Ψ): {norm_cut:.4f} | Intraconnectivity (α): {intraconn:.4f}</span>',
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=100, b=20)
    )
    
    return fig

def create_all_communities_overview(communities, G, overlapping_nodes):
    """Create overview visualization showing all communities with overlap"""
    # Use different colors for each community
    colors = px.colors.qualitative.Set3[:max(12, len(communities))]
    if len(communities) > len(colors):
        colors = colors * (len(communities) // len(colors) + 1)
    
    fig = go.Figure()
    
    # Create a layout for all nodes
    all_nodes = set()
    for comm in communities:
        all_nodes.update(comm)
    
    if not all_nodes:
        fig.add_annotation(text="Tidak ada komunitas untuk divisualisasi", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Create subgraph with all community nodes
    subgraph = G.subgraph(all_nodes)
    
    # Use force-directed layout with communities
    pos = nx.spring_layout(subgraph, k=3.0, iterations=100, seed=42)
    
    # Draw edges first (in background)
    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='rgba(125, 125, 125, 0.2)'),
        hoverinfo='none',
        showlegend=False,
        name='Edges'
    ))
    
    # Track which nodes have been drawn
    drawn_nodes = set()
    
    # First, draw non-overlapping nodes with their community colors
    for idx, community in enumerate(communities):
        # Only draw non-overlapping nodes from this community
        non_overlap_nodes = [n for n in community if n not in overlapping_nodes and n not in drawn_nodes]
        
        if non_overlap_nodes:
            node_x = [pos[node][0] for node in non_overlap_nodes if node in pos]
            node_y = [pos[node][1] for node in non_overlap_nodes if node in pos]
            node_text = [f"{node}<br>Komunitas {idx+1}<br>Degree: {subgraph.degree(node)}" 
                        for node in non_overlap_nodes if node in pos]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=colors[idx % len(colors)],
                    line=dict(width=2, color='black')
                ),
                text=[n for n in non_overlap_nodes if n in pos],
                textposition="top center",
                textfont=dict(size=8),
                hovertext=node_text,
                hoverinfo='text',
                name=f'Komunitas {idx+1}',
                legendgroup=f'comm_{idx}',
                showlegend=True if idx < 10 else False  # Limit legend items
            ))
            
            drawn_nodes.update(non_overlap_nodes)
    
    # Draw overlapping nodes with multi-color outlines
    overlap_nodes_list = [n for n in overlapping_nodes if n in all_nodes and n in pos]
    
    if overlap_nodes_list:
        # For each overlapping node, create multiple concentric circles
        for node in overlap_nodes_list:
            x, y = pos[node]
            
            # Find which communities this node belongs to
            node_communities = [i for i, comm in enumerate(communities) if node in comm]
            comm_colors = [colors[i % len(colors)] for i in node_communities]
            
            # Create hover text
            comm_indices = [i+1 for i in node_communities]
            hover_text = f"{node}<br>Overlap Protein<br>Komunitas: {comm_indices}<br>Degree: {subgraph.degree(node)}"
            
            # Base size for nodes
            base_size = 20
            
            # Draw concentric circles for each community (outer to inner)
            for i in range(len(node_communities)-1, -1, -1):
                circle_size = base_size + (i * 5)  # Each outer circle is 5 pixels larger
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    marker=dict(
                        size=circle_size,
                        color='rgba(255,255,255,0)',  # Transparent fill
                        line=dict(width=3, color=comm_colors[i])
                    ),
                    showlegend=False,
                    hoverinfo='skip',
                    legendgroup='overlap'
                ))
            
            # Add white center fill
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(
                    size=base_size-5,
                    color='white',
                    line=dict(width=1, color='black')
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add text label
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='text',
                text=[node],
                textposition="top center",
                textfont=dict(size=9, color='black', family='Arial Black'),
                hovertext=[hover_text],
                hoverinfo='text',
                showlegend=False
            ))
        
        # Add legend entry showing example of multi-outline
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=20,
                color='white',
                line=dict(width=3, color='black')
            ),
            name='Overlap (Multi-outline)',
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Visualisasi Semua Komunitas</b><br>' +
                 f'<span style="font-size:14px">Total: {len(communities)} komunitas, {len(all_nodes)} protein, {len(overlapping_nodes)} overlap protein</span>',
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=700,
        legend=dict(
            x=1.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(l=20, r=150, t=100, b=20)
    )
    
    # Add annotation about overlapping nodes
    fig.add_annotation(
        text="⭕ = Single Community | ⭕⭕ = Multiple Communities (each outline = different community)",
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        xanchor='left', yanchor='bottom',
        showarrow=False,
        font=dict(size=12, color='#2c3e50'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#2c3e50',
        borderwidth=1
    )
    
    return fig

# Callback for exporting communities to CSV
@app.callback(
    Output("download-communities-csv", "data"),
    Output("export-store", "data"),
    Input("export-csv-btn", "n_clicks"),
    State("export-store", "data"),
    prevent_initial_call=True
)
def export_communities_csv(n_clicks, store_data):
    """Export community detection results with internal density metrics to CSV file"""
    global current_results, detector
    
    # Initialize store data if None
    if store_data is None:
        store_data = {"clicks": 0}
    
    # Only proceed if this is a NEW click (n_clicks increased)
    previous_clicks = store_data.get("clicks", 0)
    
    if not n_clicks or n_clicks <= previous_clicks or current_results is None or detector is None:
        print(f"❌ Export diblokir: n_clicks={n_clicks}, previous={previous_clicks}, has_results={current_results is not None}")
        return dash.no_update, {"clicks": n_clicks or 0}
    
    print(f"✅ Export dipicu: klik baru terdeteksi ({n_clicks} > {previous_clicks})")
    
    try:
        communities = current_results['communities']
        analysis = current_results['analysis']
        threshold = current_results['threshold']
        overlapping_nodes = analysis.get('overlapping_nodes', {})
        
        # Prepare comprehensive data for export with internal density as primary metric
        export_data = []
        
        for i, community in enumerate(communities):
            community_list = sorted(list(community))
            intraconn = detector.intraconnectivity(community)
            
            # Calculate normalized node cut (PRIMARY METRIC)
            normalized_node_cut = detector.normalized_node_cut(community)
            
            # Check overlap status
            has_overlap = any(node in overlapping_nodes for node in community)
            overlap_type = "Overlap" if has_overlap else "Non-overlap"
            
            # Count overlap proteins
            overlap_count = sum(1 for node in community if node in overlapping_nodes)
            overlap_proteins = [node for node in community if node in overlapping_nodes]
            
            export_data.append({
                'ID_Komunitas': i + 1,
                'Tipe': overlap_type,
                'Ukuran': len(community),
                'Jumlah_Protein_Overlap': overlap_count,
                'Normalized_Node_Cut_Psi': round(normalized_node_cut, 6),  # PRIMARY
                'Intraconnectivity_Alpha': round(intraconn, 6),  # SECONDARY
                'Semua_Protein': '; '.join(community_list),
                'Protein_Overlap': '; '.join(overlap_proteins) if overlap_proteins else 'Tidak ada',
                'APAL_Threshold_Digunakan': threshold
            })
        
        # Create DataFrame
        df = pd.DataFrame(export_data)
        
        # Add metadata header information with normalized node cut emphasis (UNWEIGHTED)
        metadata = {
            'Waktu_Ekspor': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Total_Komunitas': len(communities),
            'Komunitas_Overlap': sum(1 for row in export_data if row['Tipe'] == 'Overlap'),
            'APAL_Threshold': threshold,
            'Metrik_Utama': 'Normalized_Node_Cut_Havemann_et_al_2012_UNWEIGHTED',
            'Node_Jaringan': len(detector.graph.nodes()),
            'Edge_Jaringan': len(detector.graph.edges()),
            'Algoritma': 'APAL - Adjacency Propagation Algorithm',
            'Metode_Evaluasi': 'Normalized Node Cut UNWEIGHTED (Havemann et al. 2012)',
            'Sumber_Data': 'File yang diunggah pengguna',
            'Penggunaan_Bobot': 'Unweighted graph (tidak menggunakan bobot)'
        }
        
        # Create final export with metadata emphasizing UNWEIGHTED normalized node cut
        csv_string = "# Hasil Deteksi Komunitas APAL dengan Evaluasi UNWEIGHTED Normalized Node Cut\n"
        csv_string += "# Adjacency Propagation Algorithm untuk Deteksi Komunitas Overlap\n"
        csv_string += "# Dievaluasi menggunakan UNWEIGHTED Normalized Node Cut (Havemann et al. 2012)\n"
        csv_string += "# Dihasilkan oleh Antarmuka Dash APAL\n"
        csv_string += "#\n"
        for key, value in metadata.items():
            csv_string += f"# {key}: {value}\n"
        csv_string += "#\n"
        csv_string += "# Metrik Evaluasi Utama: UNWEIGHTED Normalized Node Cut (Ψ)\n"
        csv_string += "# Formula: Ψ = (1/k_in) * Σ[(k_in_i * k_out_i)/k_i] - HANYA MENGHITUNG EDGE, TANPA BOBOT\n"
        csv_string += "# Referensi: Havemann et al. (2012) 'Evaluating Overlapping Communities with the Conductance of their Boundary Nodes'\n"
        csv_string += "# Implementasi: UNWEIGHTED sesuai Section 2.1 halaman 2 paper\n"
        csv_string += "#\n"
        csv_string += "# Deskripsi Kolom:\n"
        csv_string += "# ID_Komunitas: Pengenal unik untuk setiap komunitas\n"
        csv_string += "# Tipe: Overlap (berbagi protein dengan komunitas lain) atau Non-overlap\n"
        csv_string += "# Ukuran: Jumlah protein dalam komunitas\n"
        csv_string += "# Jumlah_Protein_Overlap: Jumlah protein yang termasuk dalam beberapa komunitas\n"
        csv_string += "# Normalized_Node_Cut_Psi: Ψ = metrik conductance UNWEIGHTED - METRIK KUALITAS UTAMA (semakin rendah = semakin baik)\n"
        csv_string += "# Intraconnectivity_Alpha: α = k/[n(n-1)] - metrik algoritma APAL\n"
        csv_string += "# Semua_Protein: Daftar lengkap protein dalam komunitas\n"
        csv_string += "# Protein_Overlap: Daftar protein overlap (anggota multi-komunitas)\n"
        csv_string += "#\n"
        
        # Add the actual data
        csv_string += df.to_csv(index=False)
        
        filename = f"Komunitas_APAL_NormalizedCut_t{threshold}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        print(f"✅ Exporting CSV with normalized node cut metrics: {filename}")
        return dict(content=csv_string, filename=filename), {"clicks": n_clicks}
        
    except Exception as e:
        print(f"❌ Error exporting CSV: {str(e)}")
        return dash.no_update, {"clicks": n_clicks or 0}

def create_community_details_table(communities, detector, overlapping_nodes, threshold):
    """Create simplified community table with normalized node cut as primary metric"""
    if not communities:
        return html.Div("Tidak ditemukan komunitas.", style={'textAlign': 'center', 'color': '#7f8c8d'})
    
    # Calculate normalized node cuts (primary quality metric)
    normalized_node_cuts = detector.calculate_normalized_node_cuts(communities)
    
    # Prepare data for table
    table_data = []
    for i, community in enumerate(communities):
        community_list = sorted(list(community))
        intraconn = detector.intraconnectivity(community)
        
        # Calculate normalized node cut - PRIMARY METRIC (UNWEIGHTED)
        normalized_cut = detector.normalized_node_cut(community)
        
        # Check if community has overlapping nodes
        has_overlap = any(node in overlapping_nodes for node in community)
        overlap_type = "🔄 Overlap" if has_overlap else "⚪ Non-overlap"
        
        # Count overlap proteins in this community
        overlap_count = sum(1 for node in community if node in overlapping_nodes)
        
        table_data.append({
            'ID Komunitas': i + 1,
            'Tipe': overlap_type,
            'Ukuran': len(community),
            'Overlap Protein': overlap_count,
            'Normalized Cut (Ψ)': f"{normalized_cut:.4f}",
            'Intraconnectivity (α)': f"{intraconn:.4f}",
            'Contoh Protein': ', '.join(community_list[:5]) + ('...' if len(community_list) > 5 else '')
        })
    
    # Create data table
    table = dash_table.DataTable(
        data=table_data,
        columns=[
            {'name': 'ID', 'id': 'ID Komunitas', 'type': 'numeric'},
            {'name': 'Tipe', 'id': 'Tipe', 'type': 'text'},
            {'name': 'Ukuran', 'id': 'Ukuran', 'type': 'numeric'},
            {'name': 'Overlap Protein', 'id': 'Overlap Protein', 'type': 'numeric'},
            {'name': 'Ψ (Normalized Cut)', 'id': 'Normalized Cut (Ψ)', 'type': 'numeric'},
            {'name': 'α (Intraconnectivity)', 'id': 'Intraconnectivity (α)', 'type': 'numeric'},
            {'name': 'Contoh Protein', 'id': 'Contoh Protein', 'type': 'text'}
        ],
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
        style_data={'backgroundColor': '#f8f9fa'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{Tipe} contains "Overlap"'},
                'backgroundColor': '#fff3cd',
                'color': 'black',
            }
        ],
        sort_action='native',
        page_size=15
    )
    
    return html.Div([
        html.H4("Detail Komunitas", style={'color': '#2c3e50', 'marginBottom': '20px'}),
        table
    ])

def create_community_size_chart(communities):
    """Create MODERN community size distribution - Lollipop style"""
    sizes = sorted([len(c) for c in communities], reverse=True)
    community_ids = [f"C{i+1}" for i in range(len(communities))]
    
    # Sort by size
    sorted_data = sorted(zip(community_ids, sizes), key=lambda x: x[1], reverse=True)
    community_ids = [x[0] for x in sorted_data]
    sizes = [x[1] for x in sorted_data]
    
    # Limit to top 20 for readability
    if len(sizes) > 20:
        community_ids = community_ids[:20]
        sizes = sizes[:20]
    
    fig = go.Figure()
    
    # Gradient colors based on size
    max_size = max(sizes) if sizes else 1
    colors = [f'rgba(99, 102, 241, {0.4 + 0.6 * (s / max_size)})' for s in sizes]
    
    # Horizontal bar chart (cleaner look)
    fig.add_trace(go.Bar(
        y=community_ids,
        x=sizes,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=sizes,
        textposition='outside',
        textfont=dict(size=11, color='#6366f1'),
        hovertemplate='<b>%{y}</b><br>Ukuran: %{x} protein<extra></extra>'
    ))
    
    avg_size = np.mean(sizes)
    fig.add_vline(x=avg_size, line=dict(dash="dot", color='#f59e0b', width=2))
    fig.add_annotation(
        x=avg_size, y=1.05, yref='paper',
        text=f"Avg: {avg_size:.1f}", showarrow=False,
        font=dict(size=10, color='#f59e0b')
    )
    
    fig.update_layout(
        title=dict(
            text='<b>Ukuran Komunitas</b>',
            font=dict(size=15, color='#1e293b'),
            x=0, xanchor='left'
        ),
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=False,
            tickfont=dict(size=10, color='#94a3b8')
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(size=10, color='#64748b'),
            autorange='reversed'
        ),
        height=max(280, len(sizes) * 22 + 80),
        margin=dict(l=50, r=60, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(bgcolor='white', font_size=12, font_color='black', bordercolor='#e2e8f0')
    )
    
    return fig

def create_network_stats(G):
    """Create MODERN network statistics display"""
    if G is None or G.number_of_nodes() == 0:
        return html.Div("Data jaringan tidak tersedia", style={'color': '#64748b', 'textAlign': 'center', 'padding': '20px'})
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    max_possible_edges = (num_nodes * (num_nodes - 1)) / 2
    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
    avg_degree = np.mean([G.degree(n) for n in G.nodes()])
    
    def stat_card(value, label, color):
        return html.Div([
            html.Div(str(value), style={
                'fontSize': '28px', 'fontWeight': '700', 'color': color, 'lineHeight': '1'
            }),
            html.Div(label, style={
                'fontSize': '11px', 'color': '#64748b', 'marginTop': '4px', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'
            })
        ], style={
            'textAlign': 'center', 'padding': '16px 12px', 'backgroundColor': '#f8fafc',
            'borderRadius': '10px', 'flex': '1', 'minWidth': '100px'
        })
    
    return html.Div([
        html.Div([
            stat_card(f"{num_nodes:,}", "Nodes", '#6366f1'),
            stat_card(f"{num_edges:,}", "Edges", '#8b5cf6'),
            stat_card(f"{avg_degree:.1f}", "Avg Degree", '#06b6d4'),
            stat_card(f"{density:.4f}", "Density", '#10b981'),
        ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap'})
    ])

def create_network_visualization(G):
    """Create MODERN network visualization with degree-based coloring"""
    fig = go.Figure()
    
    if G is None or G.number_of_nodes() == 0:
        fig.add_annotation(
            text="Upload data untuk melihat visualisasi",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color='#94a3b8')
        )
        fig.update_layout(
            height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False)
        )
        return fig
    
    # Sample nodes for performance
    nodes_sample = list(G.nodes())[:100] if G.number_of_nodes() > 100 else list(G.nodes())
    G_sample = G.subgraph(nodes_sample)
    
    if G_sample.number_of_nodes() > 0:
        pos = nx.spring_layout(G_sample, k=1.5, iterations=50, seed=42)
        
        # Edges with gradient opacity
        edge_x, edge_y = [], []
        for edge in G_sample.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.6, color='rgba(99, 102, 241, 0.15)'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Nodes with degree-based sizing and coloring
        degrees = [G_sample.degree(node) for node in G_sample.nodes()]
        max_deg = max(degrees) if degrees else 1
        node_sizes = [8 + 12 * (d / max_deg) for d in degrees]
        
        node_x = [pos[node][0] for node in G_sample.nodes()]
        node_y = [pos[node][1] for node in G_sample.nodes()]
        node_text = [f"<b>{node}</b><br>Degree: {G_sample.degree(node)}" for node in G_sample.nodes()]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=degrees,
                colorscale=[[0, '#c7d2fe'], [0.5, '#6366f1'], [1, '#312e81']],
                showscale=True,
                colorbar=dict(
                    title=dict(text='Degree', font=dict(size=10, color='#64748b')),
                    thickness=12,
                    len=0.5,
                    tickfont=dict(size=9, color='#94a3b8'),
                    x=1.02
                ),
                line=dict(width=1.5, color='white')
            ),
            text=node_text,
            hoverinfo='text',
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>Network Preview</b> ({len(nodes_sample)} nodes)',
            font=dict(size=15, color='#1e293b'),
            x=0, xanchor='left'
        ),
        height=300,
        margin=dict(l=10, r=60, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hoverlabel=dict(bgcolor='white', font_size=12, font_color='black', bordercolor='#e2e8f0')
    )
    
    return fig

def create_degree_distribution(G):
    """Create MODERN degree distribution - Area chart style"""
    fig = go.Figure()
    
    if G is None or G.number_of_nodes() == 0:
        fig.add_annotation(
            text="Upload data untuk melihat distribusi",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color='#94a3b8')
        )
        fig.update_layout(
            height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False)
        )
        return fig
    
    degrees = [G.degree(n) for n in G.nodes()]
    
    if degrees:
        # Create smooth histogram data
        from collections import Counter
        degree_counts = Counter(degrees)
        x_vals = sorted(degree_counts.keys())
        y_vals = [degree_counts[x] for x in x_vals]
        
        # Area chart
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#8b5cf6', width=2, shape='spline'),
            fillcolor='rgba(139, 92, 246, 0.2)',
            hovertemplate='<b>Degree: %{x}</b><br>Nodes: %{y}<extra></extra>'
        ))
        
        # Average marker
        avg_degree = np.mean(degrees)
        fig.add_vline(x=avg_degree, line=dict(dash="dot", color='#f59e0b', width=2))
        fig.add_annotation(
            x=avg_degree, y=1.05, yref='paper',
            text=f"Avg: {avg_degree:.1f}", showarrow=False,
            font=dict(size=10, color='#f59e0b')
        )
        
        # Max degree annotation
        max_deg = max(degrees)
        fig.add_annotation(
            x=max_deg, y=degree_counts[max_deg],
            text=f"Max: {max_deg}", showarrow=True, arrowhead=2,
            ax=30, ay=-20, font=dict(size=10, color='#6366f1')
        )
    
    fig.update_layout(
        title=dict(
            text='<b>Distribusi Degree</b>',
            font=dict(size=15, color='#1e293b'),
            x=0, xanchor='left'
        ),
        xaxis=dict(
            title=dict(text='Degree', font=dict(size=11, color='#94a3b8')),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=False,
            tickfont=dict(size=10, color='#94a3b8')
        ),
        yaxis=dict(
            title=dict(text='Jumlah Node', font=dict(size=11, color='#94a3b8')),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)',
            zeroline=False,
            tickfont=dict(size=10, color='#94a3b8')
        ),
        height=300,
        margin=dict(l=50, r=30, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(bgcolor='white', font_size=12, font_color='black', bordercolor='#e2e8f0')
    )
    
    return fig

# Custom CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f5f6fa;
                margin: 0;
                padding: 20px;
            }
            .dash-table-container {
                max-height: 400px;
                overflow-y: auto;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)