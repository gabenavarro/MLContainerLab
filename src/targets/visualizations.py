"""
Protein Network Visualization Module
====================================
Creates publication-quality visualizations of protein interaction networks.

Features:
- Network graph visualization with distance-based coloring
- Interactive plots using plotly
- Static plots using matplotlib
- Heatmap of interaction distances


def visualize_results(analyzer, results_df: pd.DataFrame, protein_list: List[str]):
    '''
    Convenience function to create all visualizations.
    
    Parameters:
        analyzer: ProteinNetworkAnalyzer instance
        results_df (pd.DataFrame): Analysis results
        protein_list (List[str]): List of query proteins
    '''
    visualizer = ProteinNetworkVisualizer(analyzer)
    
    print("\nGenerating visualizations...")
    
    # Create static network plot
    visualizer.plot_network_static(
        protein_list,
        save_path=f"{analyzer.target_protein}_network_static.png"
    )
    
    # Create distance heatmap
    visualizer.plot_distance_heatmap(
        results_df,
        save_path=f"{analyzer.target_protein}_distances.png"
    )
    
    # Create summary report
    visualizer.create_summary_report(
        results_df,
        save_path=f"{analyzer.target_protein}_report.png"
    )
    
    # Create interactive plot if plotly available
    if PLOTLY_AVAILABLE:
        visualizer.plot_network_interactive(
            protein_list,
            save_path=f"{analyzer.target_protein}_network_interactive.html"
        )
    
    print("\nAll visualizations created successfully!")


"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from typing import Optional, Dict, List
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive plots will be disabled.")


class ProteinNetworkVisualizer:
    """
    Visualization tools for protein interaction networks.
    """
    
    def __init__(self, analyzer):
        """
        Initialize visualizer with a ProteinNetworkAnalyzer instance.
        
        Parameters:
            analyzer: ProteinNetworkAnalyzer instance with built network
        """
        self.analyzer = analyzer
        self.graph = analyzer.graph
        self.target_protein = analyzer.target_protein
        
    def plot_network_static(self, 
                          protein_list: List[str],
                          figsize: tuple = (14, 10),
                          save_path: Optional[str] = None) -> None:
        """
        Create static network visualization using matplotlib.
        
        Parameters:
            protein_list (List[str]): Proteins to highlight in the network
            figsize (tuple): Figure size
            save_path (str): Path to save figure (optional)
        """
        if self.graph.number_of_nodes() == 0:
            print("Error: Network is empty. Build network first.")
            return
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Calculate layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # Calculate distances from target for coloring
        distances = {}
        protein_list_upper = [p.upper() for p in protein_list]
        
        for node in self.graph.nodes():
            try:
                if nx.has_path(self.graph, self.target_protein, node):
                    distances[node] = nx.shortest_path_length(
                        self.graph, self.target_protein, node
                    )
                else:
                    distances[node] = 999
            except:
                distances[node] = 999
        
        # Create color map based on distance
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        for node in self.graph.nodes():
            if node == self.target_protein:
                node_colors.append('#FF0000')  # Red for target
                node_sizes.append(1000)
                node_labels[node] = node
            elif node.upper() in protein_list_upper:
                dist = distances.get(node, 999)
                if dist <= 2:
                    node_colors.append('#00AA00')  # Green for close proteins
                else:
                    node_colors.append('#FFA500')  # Orange for distant proteins
                node_sizes.append(700)
                node_labels[node] = node
            else:
                node_colors.append('#87CEEB')  # Light blue for intermediate
                node_sizes.append(300)
                # Only label if it's a hub
                if self.graph.degree(node) > 5:
                    node_labels[node] = node
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos, 
            alpha=0.2, 
            edge_color='gray',
            width=0.5,
            ax=ax
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos,
            labels=node_labels,
            font_size=9,
            font_weight='bold',
            ax=ax
        )
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='#FF0000', markersize=15, label='Target Protein'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='#00AA00', markersize=12, label='Query (≤2 degrees)'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='#FFA500', markersize=12, label='Query (>2 degrees)'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='#87CEEB', markersize=10, label='Intermediate')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.set_title(f'Protein Interaction Network\nTarget: {self.target_protein}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Static network plot saved to {save_path}")
        
        plt.show()
    
    def plot_network_interactive(self, 
                                protein_list: List[str],
                                save_path: Optional[str] = None) -> None:
        """
        Create interactive network visualization using plotly.
        
        Parameters:
            protein_list (List[str]): Proteins to highlight
            save_path (str): Path to save HTML file (optional)
        """
        if not PLOTLY_AVAILABLE:
            print("Error: Plotly is not installed. Cannot create interactive plot.")
            return
        
        if self.graph.number_of_nodes() == 0:
            print("Error: Network is empty. Build network first.")
            return
        
        # Calculate layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # Calculate distances
        distances = {}
        protein_list_upper = [p.upper() for p in protein_list]
        
        for node in self.graph.nodes():
            try:
                if nx.has_path(self.graph, self.target_protein, node):
                    distances[node] = nx.shortest_path_length(
                        self.graph, self.target_protein, node
                    )
                else:
                    distances[node] = 999
            except:
                distances[node] = 999
        
        # Create edge traces
        edge_x = []
        edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            dist = distances.get(node, 999)
            degree = self.graph.degree(node)
            
            # Create hover text
            text = f"<b>{node}</b><br>"
            text += f"Distance from {self.target_protein}: {dist if dist < 999 else 'Not connected'}<br>"
            text += f"Degree: {degree}"
            node_text.append(text)
            
            # Set colors and sizes
            if node == self.target_protein:
                node_color.append('red')
                node_size.append(30)
            elif node.upper() in protein_list_upper:
                if dist <= 2:
                    node_color.append('green')
                else:
                    node_color.append('orange')
                node_size.append(20)
            else:
                node_color.append('lightblue')
                node_size.append(10)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node if node == self.target_protein or node.upper() in protein_list_upper 
                  else '' for node in self.graph.nodes()],
            textposition="top center",
            textfont=dict(size=10, color='black'),
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line=dict(width=2, color='white')
            ),
            hovertext=node_text
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=f'<b>Protein Interaction Network</b><br>Target: {self.target_protein}',
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=100),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                height=800
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive network plot saved to {save_path}")
        
        fig.show()
    
    def plot_distance_heatmap(self, 
                             results_df: pd.DataFrame,
                             figsize: tuple = (10, 8),
                             save_path: Optional[str] = None) -> None:
        """
        Create heatmap showing distances and functional relationships.
        
        Parameters:
            results_df (pd.DataFrame): Results from analyze_protein_distances
            figsize (tuple): Figure size
            save_path (str): Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Filter to proteins in network with valid distances
        valid_results = results_df[
            (results_df['in_network'] == True) & 
            (results_df['distance'].notna())
        ].copy()
        
        if len(valid_results) == 0:
            print("No valid results to plot")
            return
        
        # Plot 1: Distance bar chart
        valid_results_sorted = valid_results.sort_values('distance')
        colors = ['green' if d <= 2 else 'orange' 
                 for d in valid_results_sorted['distance']]
        
        ax1.barh(range(len(valid_results_sorted)), 
                valid_results_sorted['distance'],
                color=colors, alpha=0.7)
        ax1.set_yticks(range(len(valid_results_sorted)))
        ax1.set_yticklabels(valid_results_sorted['protein'])
        ax1.set_xlabel('Degrees of Separation', fontsize=12, fontweight='bold')
        ax1.set_title(f'Distance from {self.target_protein}', 
                     fontsize=14, fontweight='bold')
        ax1.axvline(x=2, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Functional relationships
        close_proteins = valid_results[valid_results['distance'] <= 2].copy()
        
        if len(close_proteins) > 0:
            func_rel_counts = {
                'activator': 0,
                'inhibitor': 0,
                'unknown': 0
            }
            
            for _, row in close_proteins.iterrows():
                rel = str(row['functional_relationship']).lower()
                if 'activat' in rel:
                    func_rel_counts['activator'] += 1
                elif 'inhibit' in rel:
                    func_rel_counts['inhibitor'] += 1
                else:
                    func_rel_counts['unknown'] += 1
            
            colors_pie = ['#2ecc71', '#e74c3c', '#95a5a6']
            ax2.pie(func_rel_counts.values(), 
                   labels=func_rel_counts.keys(),
                   autopct='%1.1f%%',
                   colors=colors_pie,
                   startangle=90)
            ax2.set_title('Functional Relationships\n(≤2 degrees)', 
                         fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No proteins within\n2 degrees', 
                    ha='center', va='center', fontsize=14)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distance heatmap saved to {save_path}")
        
        plt.show()
    
    def create_summary_report(self, 
                            results_df: pd.DataFrame,
                            save_path: str = None) -> None:
        """
        Create a comprehensive visual summary report.
        
        Parameters:
            results_df (pd.DataFrame): Analysis results
            save_path (str): Path to save report
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Network statistics
        ax1 = fig.add_subplot(gs[0, :])
        stats = self.analyzer.get_network_statistics()
        stats_text = f"""
        NETWORK STATISTICS
        Target Protein: {self.target_protein}
        Nodes: {stats.get('num_nodes', 'N/A')} | Edges: {stats.get('num_edges', 'N/A')}
        Density: {stats.get('density', 0):.4f} | Connected: {stats.get('is_connected', 'N/A')}
        Target Connections: {stats.get('target_num_connections', 'N/A')}
        """
        ax1.text(0.5, 0.5, stats_text, ha='center', va='center', 
                fontsize=12, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax1.axis('off')
        
        # Distance distribution
        ax2 = fig.add_subplot(gs[1, :2])
        valid_results = results_df[results_df['distance'].notna()].copy()
        if len(valid_results) > 0:
            distances = valid_results['distance'].values
            distances = distances[distances != float('inf')]
            if len(distances) > 0:
                ax2.hist(distances, bins=range(int(max(distances))+2), 
                        color='steelblue', alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Degrees of Separation', fontweight='bold')
                ax2.set_ylabel('Count', fontweight='bold')
                ax2.set_title('Distance Distribution', fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
        
        # Functional relationships
        ax3 = fig.add_subplot(gs[1, 2])
        close_proteins = results_df[results_df['distance'] <= 2]
        if len(close_proteins) > 0:
            activators = len(close_proteins[close_proteins['functional_relationship'].astype(str).str.contains('activat', case=False, na=False)])
            inhibitors = len(close_proteins[close_proteins['functional_relationship'].astype(str).str.contains('inhibit', case=False, na=False)])
            unknown = len(close_proteins) - activators - inhibitors
            
            ax3.bar(['Activator', 'Inhibitor', 'Unknown'], 
                   [activators, inhibitors, unknown],
                   color=['#2ecc71', '#e74c3c', '#95a5a6'])
            ax3.set_ylabel('Count', fontweight='bold')
            ax3.set_title('Functional Roles\n(≤2 degrees)', fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
        
        # Results table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = results_df[['protein', 'distance', 'functional_relationship']].head(10)
        table = ax4.table(cellText=table_data.values, 
                         colLabels=table_data.columns,
                         cellLoc='left',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(table_data.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle(f'Protein Network Analysis Report: {self.target_protein}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary report saved to {save_path}")
        
        plt.show()

