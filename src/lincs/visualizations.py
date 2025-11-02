"""
Publication-Quality Visualization Suite for LINCS BAI Analysis
================================================================

Creates journal-ready figures for Bayesian Agreement Index (BAI) analysis results.
Compatible with Nature, Science, Cell, PLOS Biology requirements.

Requirements:
- matplotlib>=3.7.0
- seaborn>=0.12.0
- numpy>=1.24.0
- pandas>=2.0.0
- scipy>=1.10.0



# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Simulate BAI results
    n_compounds = 500
    synthetic_results = pd.DataFrame({
        'inchi_key': [f'COMPOUND_{i:04d}' for i in range(n_compounds)],
        'bai_score': np.random.beta(5, 2, n_compounds),
        'n_replicates': np.random.randint(2, 8, n_compounds),
        'mean_agreement': np.random.beta(6, 3, n_compounds),
        'cell': np.random.choice(['A375', 'MCF7', 'HepG2'], n_compounds),
        'dose': np.random.choice(['1uM', '10uM', '100uM'], n_compounds),
    })
    
    # Add credible intervals
    ci_width = 0.15 / np.sqrt(synthetic_results['n_replicates'])
    synthetic_results['lower_credible'] = np.clip(
        synthetic_results['bai_score'] - ci_width, 0, 1
    )
    synthetic_results['upper_credible'] = np.clip(
        synthetic_results['bai_score'] + ci_width, 0, 1
    )
    
    # Create visualizations
    print("Creating comprehensive BAI analysis visualizations...")
    
    visualizer = BAIVisualizer(synthetic_results)
    visualizer.create_comprehensive_report('bai_comprehensive_report.png')
    
    print("\nVisualization suite complete!")
    print("Generated files:")
    print("  - bai_comprehensive_report.png: Multi-panel analysis figure")
    print("\nFor additional plots, use standalone functions:")
    print("  - plot_gene_agreement_profile()")
    print("  - plot_replicate_consistency()")
    print("  - plot_heatmap_concordance()")
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, List, Tuple, Dict
import warnings

# Publication settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color schemes for accessibility
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#06A77D',      # Green
    'warning': '#D62839',      # Red
    'gray': '#6C757D',         # Gray
}

# Colorblind-friendly palettes
PALETTE_CATEGORICAL = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
PALETTE_SEQUENTIAL = sns.color_palette("YlGnBu", n_colors=10)


class BAIVisualizer:
    """
    Comprehensive visualization suite for BAI analysis results.
    
    Attributes
    ----------
    results_df : pd.DataFrame
        BAI analysis results with columns: bai_score, lower_credible, 
        upper_credible, mean_agreement, n_replicates, plus metadata
    """
    
    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize visualizer with results.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results from compute_signature_concordance()
        """
        self.results = results_df.copy()
        self._validate_results()
        
    def _validate_results(self):
        """Validate required columns exist."""
        required = ['bai_score', 'lower_credible', 'upper_credible', 'mean_agreement']
        missing = [col for col in required if col not in self.results.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def create_comprehensive_report(
        self, 
        output_path: str = 'bai_analysis_report.png',
        figsize: Tuple[float, float] = (11, 8.5)
    ):
        """
        Create multi-panel comprehensive report figure.
        
        Parameters
        ----------
        output_path : str
            Path to save figure
        figsize : tuple
            Figure size in inches (width, height)
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)
        
        # Panel A: BAI Score Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_bai_distribution(ax1)
        ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes, 
                fontsize=12, fontweight='bold', va='top')
        
        # Panel B: Credible Interval Plot
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_credible_intervals(ax2)
        ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes, 
                fontsize=12, fontweight='bold', va='top')
        
        # Panel C: Agreement vs BAI
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_agreement_correlation(ax3)
        ax3.text(-0.15, 1.05, 'C', transform=ax3.transAxes, 
                fontsize=12, fontweight='bold', va='top')
        
        # Panel D: Top Concordant Hits
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_top_hits(ax4, top_n=20)
        ax4.text(-0.08, 1.05, 'D', transform=ax4.transAxes, 
                fontsize=12, fontweight='bold', va='top')
        
        # Panel E: Replicate Number Effect
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_replicate_effect(ax5)
        ax5.text(-0.15, 1.05, 'E', transform=ax5.transAxes, 
                fontsize=12, fontweight='bold', va='top')
        
        # Panel F: Uncertainty Distribution
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_uncertainty_distribution(ax6)
        ax6.text(-0.15, 1.05, 'F', transform=ax6.transAxes, 
                fontsize=12, fontweight='bold', va='top')
        
        # Panel G: Concordance Threshold Analysis
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_threshold_analysis(ax7)
        ax7.text(-0.15, 1.05, 'G', transform=ax7.transAxes, 
                fontsize=12, fontweight='bold', va='top')
        
        # Panel H: Summary Statistics
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_summary_stats(ax8)
        ax8.text(-0.15, 1.05, 'H', transform=ax8.transAxes, 
                fontsize=12, fontweight='bold', va='top')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive report saved to: {output_path}")
        return fig
    
    def _plot_bai_distribution(self, ax):
        """Plot BAI score distribution with kernel density estimate."""
        bai_scores = self.results['bai_score'].values
        
        # Histogram
        ax.hist(bai_scores, bins=50, alpha=0.6, color=COLORS['primary'], 
                edgecolor='black', linewidth=0.5, density=True, label='Observed')
        
        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(bai_scores)
        x_range = np.linspace(0, 1, 200)
        ax.plot(x_range, kde(x_range), color=COLORS['secondary'], 
                linewidth=2, label='KDE')
        
        # Mean and median lines
        mean_val = bai_scores.mean()                                            # type: ignore
        median_val = np.median(bai_scores)                                      # type: ignore
        ax.axvline(mean_val, color=COLORS['warning'], linestyle='--', 
                   linewidth=1.5, label=f'Mean = {mean_val:.3f}')
        ax.axvline(median_val, color=COLORS['success'], linestyle='--', 
                   linewidth=1.5, label=f'Median = {median_val:.3f}')
        
        ax.set_xlabel('BAI Score', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.set_title('BAI Score Distribution', fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, frameon=False)
        ax.set_xlim(0, 1)
        sns.despine(ax=ax)
    
    def _plot_credible_intervals(self, ax, n_display: int = 30):
        """Plot credible intervals for top concordant compounds."""
        # Sort by BAI score and take top N
        top_results = self.results.nlargest(n_display, 'bai_score').copy()
        top_results = top_results.sort_values('bai_score')
        
        y_pos = np.arange(len(top_results))
        
        # Plot intervals
        for i, (idx, row) in enumerate(top_results.iterrows()):
            # Credible interval
            ax.plot([row['lower_credible'], row['upper_credible']], 
                   [i, i], 'o-', color=COLORS['gray'], linewidth=1.5, 
                   markersize=3, alpha=0.6)
            # Point estimate
            ax.plot(row['bai_score'], i, 'o', color=COLORS['primary'], 
                   markersize=5, zorder=3)
        
        # Threshold line
        threshold = 0.7
        ax.axvline(threshold, color=COLORS['warning'], linestyle='--', 
                   linewidth=1, alpha=0.7, label=f'Threshold = {threshold}')
        
        ax.set_xlabel('BAI Score', fontsize=8)
        ax.set_ylabel('Compound Rank', fontsize=8)
        ax.set_title(f'Top {n_display} Compounds: 95% Credible Intervals', 
                    fontsize=9, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, len(top_results))
        ax.legend(fontsize=6, frameon=False, loc='lower right')
        sns.despine(ax=ax)
    
    def _plot_agreement_correlation(self, ax):
        """Plot correlation between mean agreement and BAI score."""
        x = self.results['mean_agreement'].values
        y = self.results['bai_score'].values
        
        # Scatter plot with density coloring
        from matplotlib.colors import Normalize
        from scipy.stats import gaussian_kde
        
        # Calculate point density for coloring
        xy = np.vstack([x, y])                                              # type: ignore
        z = gaussian_kde(xy)(xy)
        
        scatter = ax.scatter(x, y, c=z, s=8, alpha=0.6, cmap='viridis', 
                            edgecolors='none')
        
        # Regression line
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        line_x = np.linspace(x.min(), x.max(), 100)                         # type: ignore
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'r--', linewidth=1.5, alpha=0.8, 
               label=f'R² = {r_value**2:.3f}')                              # type: ignore
        
        # Identity line
        diag_max = min(x.max(), y.max())                                    # type: ignore
        ax.plot([0, diag_max], [0, diag_max], 'k--', linewidth=0.8, 
               alpha=0.3, label='y = x')
        
        ax.set_xlabel('Mean Gene Agreement', fontsize=8)
        ax.set_ylabel('BAI Score', fontsize=8)
        ax.set_title('Agreement vs. BAI Score', fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, frameon=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Density', fontsize=6)
        cbar.ax.tick_params(labelsize=6)
        
        sns.despine(ax=ax)
    
    def _plot_top_hits(self, ax, top_n: int = 20):
        """Plot top concordant compounds as horizontal bar chart."""
        top_results = self.results.nlargest(top_n, 'bai_score').copy()
        
        # Create compound labels (use first available identifier)
        label_cols = ['inchi_key', 'pert_iname', 'pertname', 'id']
        label_col = next((col for col in label_cols if col in top_results.columns), None)
        
        if label_col:
            # Truncate long labels
            top_results['label'] = top_results[label_col].astype(str).str[:30]
        else:
            top_results['label'] = [f'Compound {i+1}' for i in range(len(top_results))]
        
        top_results = top_results.sort_values('bai_score')
        
        # Color bars by score
        colors = plt.cm.RdYlGn(top_results['bai_score'].values) # type: ignore
        
        bars = ax.barh(range(len(top_results)), top_results['bai_score'], 
                      color=colors, edgecolor='black', linewidth=0.5)
        
        # Error bars (credible intervals)
        errors_lower = top_results['bai_score'] - top_results['lower_credible']
        errors_upper = top_results['upper_credible'] - top_results['bai_score']
        ax.errorbar(top_results['bai_score'], range(len(top_results)), 
                   xerr=[errors_lower, errors_upper], fmt='none', 
                   ecolor='black', elinewidth=0.8, capsize=2, alpha=0.6)
        
        ax.set_yticks(range(len(top_results)))
        ax.set_yticklabels(top_results['label'], fontsize=6)
        ax.set_xlabel('BAI Score', fontsize=8)
        ax.set_title(f'Top {top_n} Concordant Compounds', fontsize=9, fontweight='bold')
        ax.set_xlim(0, 1.05)
        
        # Add score annotations
        for i, (idx, row) in enumerate(top_results.iterrows()):
            ax.text(row['bai_score'] + 0.02, i, f"{row['bai_score']:.3f}", 
                   va='center', fontsize=5)
        
        sns.despine(ax=ax)
    
    def _plot_replicate_effect(self, ax):
        """Plot effect of replicate number on BAI uncertainty."""
        if 'n_replicates' not in self.results.columns:
            ax.text(0.5, 0.5, 'n_replicates column not found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return
        
        # Calculate CI width
        self.results['ci_width'] = (self.results['upper_credible'] - 
                                    self.results['lower_credible'])
        
        # Group by replicate number
        grouped = self.results.groupby('n_replicates').agg({
            'ci_width': ['mean', 'std'],
            'bai_score': 'count'
        }).reset_index()
        
        grouped.columns = ['n_replicates', 'mean_width', 'std_width', 'count']
        
        # Plot mean CI width vs n_replicates
        ax.errorbar(grouped['n_replicates'], grouped['mean_width'], 
                   yerr=grouped['std_width'], fmt='o-', color=COLORS['primary'],
                   linewidth=2, markersize=6, capsize=4, capthick=1.5)
        
        # Theoretical 1/sqrt(n) decay
        n_range = np.linspace(grouped['n_replicates'].min(), 
                             grouped['n_replicates'].max(), 100)
        theoretical = grouped['mean_width'].iloc[0] * np.sqrt(grouped['n_replicates'].iloc[0]) / np.sqrt(n_range)
        ax.plot(n_range, theoretical, 'r--', linewidth=1.5, alpha=0.7,
               label='Theoretical (∝ 1/√n)')
        
        ax.set_xlabel('Number of Replicates', fontsize=8)
        ax.set_ylabel('Mean CI Width', fontsize=8)
        ax.set_title('Uncertainty vs. Replicate Number', fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, frameon=False)
        sns.despine(ax=ax)
    
    def _plot_uncertainty_distribution(self, ax):
        """Plot distribution of credible interval widths."""
        ci_width = (self.results['upper_credible'] - 
                   self.results['lower_credible'])
        
        # Violin plot
        parts = ax.violinplot([ci_width], positions=[0], widths=0.7,
                             showmeans=True, showmedians=True)
        
        # Color violin
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS['primary'])
            pc.set_alpha(0.6)
        
        # Box plot overlay
        bp = ax.boxplot([ci_width], positions=[0], widths=0.3, 
                       patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS['accent'])
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Credible Interval Width', fontsize=8)
        ax.set_title('Uncertainty Distribution', fontsize=9, fontweight='bold')
        ax.set_xticks([])
        
        # Add statistics
        stats_text = f"Median: {np.median(ci_width):.3f}\nIQR: {stats.iqr(ci_width):.3f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=6, va='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.3))
        
        sns.despine(ax=ax, bottom=True)
    
    def _plot_threshold_analysis(self, ax):
        """Plot sensitivity to different BAI thresholds."""
        thresholds = np.linspace(0, 1, 50)
        n_concordant = [sum(self.results['bai_score'] >= t) for t in thresholds]
        
        ax.plot(thresholds, n_concordant, linewidth=2, color=COLORS['primary'])
        ax.fill_between(thresholds, 0, n_concordant, alpha=0.3, color=COLORS['primary'])
        
        # Mark common thresholds
        common_thresholds = [0.5, 0.7, 0.9]
        for t in common_thresholds:
            n = sum(self.results['bai_score'] >= t)
            ax.axvline(t, color=COLORS['warning'], linestyle='--', 
                      linewidth=1, alpha=0.5)
            ax.text(t, n, f' {n}', fontsize=6, va='bottom')
        
        ax.set_xlabel('BAI Threshold', fontsize=8)
        ax.set_ylabel('Number of Concordant Hits', fontsize=8)
        ax.set_title('Threshold Sensitivity', fontsize=9, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(self.results))
        sns.despine(ax=ax)
    
    def _plot_summary_stats(self, ax):
        """Display summary statistics table."""
        ax.axis('off')
        
        # Calculate statistics
        stats_dict = {
            'Total Compounds': len(self.results),
            'Mean BAI': f"{self.results['bai_score'].mean():.3f}",
            'Median BAI': f"{np.median(self.results['bai_score']):.3f}",
            'SD BAI': f"{self.results['bai_score'].std():.3f}",
            'High Concordance': f"{sum(self.results['bai_score'] >= 0.7)} ({100*sum(self.results['bai_score'] >= 0.7)/len(self.results):.1f}%)",
            'Mean CI Width': f"{(self.results['upper_credible'] - self.results['lower_credible']).mean():.3f}",
        }
        
        if 'n_replicates' in self.results.columns:
            stats_dict['Mean Replicates'] = f"{self.results['n_replicates'].mean():.1f}"
        
        # Create table
        table_data = [[k, v] for k, v in stats_dict.items()]
        table = ax.table(cellText=table_data, cellLoc='left',
                        colLabels=['Metric', 'Value'],
                        loc='center', bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.8)
        
        # Style table
        for i in range(len(stats_dict) + 1):
            if i == 0:  # Header
                table[(i, 0)].set_facecolor(COLORS['primary'])
                table[(i, 1)].set_facecolor(COLORS['primary'])
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                if i % 2 == 0:
                    table[(i, 0)].set_facecolor('#f0f0f0')
                    table[(i, 1)].set_facecolor('#f0f0f0')
        
        ax.set_title('Summary Statistics', fontsize=9, fontweight='bold', pad=10)


# ============================================================================
# Standalone Plotting Functions
# ============================================================================

def plot_gene_agreement_profile(
    sample_replicates: np.ndarray,
    reference_signature: np.ndarray,
    gene_names: Optional[List[str]] = None,
    top_n: int = 30,
    output_path: str = 'gene_agreement_profile.png'
):
    """
    Plot gene-wise agreement between sample and reference.
    
    Parameters
    ----------
    sample_replicates : np.ndarray, shape (n_replicates, n_genes)
        Sample replicate data
    reference_signature : np.ndarray, shape (n_genes,)
        Reference gene signature
    gene_names : list, optional
        Gene identifiers
    top_n : int
        Number of top/bottom genes to label
    output_path : str
        Output file path
    """
    n_genes = len(reference_signature)
    if gene_names is None:
        gene_names = [f'Gene_{i}' for i in range(n_genes)]
    
    # Calculate per-gene agreement
    sample_mean = np.mean(sample_replicates, axis=0)
    sample_std = np.std(sample_replicates, axis=0)
    
    # Z-scores
    pooled_std = np.sqrt(sample_std**2 + reference_signature.std()**2 + 1e-8)
    z_scores = np.abs(sample_mean - reference_signature) / pooled_std
    
    # Agreement probability
    agreement_probs = np.exp(-z_scores**2 / (2 * 0.5**2))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    # Panel A: Agreement profile across all genes
    ax = axes[0, 0]
    sorted_idx = np.argsort(agreement_probs)[::-1]
    ax.plot(range(n_genes), agreement_probs[sorted_idx], 
           color=COLORS['primary'], linewidth=1)
    ax.fill_between(range(n_genes), 0, agreement_probs[sorted_idx], 
                    alpha=0.3, color=COLORS['primary'])
    ax.axhline(0.5, color=COLORS['warning'], linestyle='--', 
              linewidth=1, alpha=0.7, label='Threshold = 0.5')
    ax.set_xlabel('Gene Rank', fontsize=8)
    ax.set_ylabel('Agreement Probability', fontsize=8)
    ax.set_title('Gene-wise Agreement Profile', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, frameon=False)
    sns.despine(ax=ax)
    
    # Panel B: Top agreeing genes
    ax = axes[0, 1]
    top_genes_idx = np.argsort(agreement_probs)[::-1][:top_n]
    y_pos = np.arange(len(top_genes_idx))
    colors_bar = plt.cm.RdYlGn(agreement_probs[top_genes_idx]) # type: ignore
    ax.barh(y_pos, agreement_probs[top_genes_idx], color=colors_bar, 
           edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([gene_names[i] for i in top_genes_idx], fontsize=6)
    ax.set_xlabel('Agreement Probability', fontsize=8)
    ax.set_title(f'Top {top_n} Agreeing Genes', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1)
    sns.despine(ax=ax)
    
    # Panel C: Z-score distribution
    ax = axes[1, 0]
    ax.hist(z_scores, bins=50, alpha=0.6, color=COLORS['primary'], 
           edgecolor='black', linewidth=0.5, density=True)
    
    # Overlay theoretical distribution
    x_range = np.linspace(0, z_scores.max(), 200)
    theoretical = stats.halfnorm.pdf(x_range, scale=1.0)
    ax.plot(x_range, theoretical, 'r--', linewidth=2, 
           label='Theoretical (null)')
    
    ax.set_xlabel('|Z-score|', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.set_title('Z-score Distribution', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, frameon=False)
    sns.despine(ax=ax)
    
    # Panel D: Sample vs Reference correlation
    ax = axes[1, 1]
    ax.scatter(reference_signature, sample_mean, s=5, alpha=0.4, 
              color=COLORS['primary'], edgecolors='none')
    
    # Regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        reference_signature, sample_mean
    )
    line_x = np.linspace(reference_signature.min(), reference_signature.max(), 100)
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'r--', linewidth=2, 
           label=f'R² = {r_value**2:.3f}\np < 0.001' if p_value < 0.001 else f'R² = {r_value**2:.3f}\np = {p_value:.3e}') # type: ignore
    
    # Identity line
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    diag_min = max(xlim[0], ylim[0])
    diag_max = min(xlim[1], ylim[1])
    ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', 
           linewidth=1, alpha=0.3, label='y = x')
    
    ax.set_xlabel('Reference Expression', fontsize=8)
    ax.set_ylabel('Sample Mean Expression', fontsize=8)
    ax.set_title('Expression Correlation', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, frameon=False)
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gene agreement profile saved to: {output_path}")
    return fig


def plot_replicate_consistency(
    sample_replicates: np.ndarray,
    gene_names: Optional[List[str]] = None,
    output_path: str = 'replicate_consistency.png'
):
    """
    Visualize consistency across technical replicates.
    
    Parameters
    ----------
    sample_replicates : np.ndarray, shape (n_replicates, n_genes)
        Replicate data
    gene_names : list, optional
        Gene identifiers
    output_path : str
        Output file path
    """
    n_replicates, n_genes = sample_replicates.shape
    
    if gene_names is None:
        gene_names = [f'Gene_{i}' for i in range(n_genes)]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Panel A: Pairwise correlation heatmap
    ax = axes[0, 0]
    corr_matrix = np.corrcoef(sample_replicates)
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n_replicates))
    ax.set_yticks(range(n_replicates))
    ax.set_xticklabels([f'Rep{i+1}' for i in range(n_replicates)], fontsize=7)
    ax.set_yticklabels([f'Rep{i+1}' for i in range(n_replicates)], fontsize=7)
    ax.set_title('Replicate Correlation Matrix', fontsize=9, fontweight='bold')
    
    # Add correlation values
    for i in range(n_replicates):
        for j in range(n_replicates):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=6)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson R', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    
    # Panel B: Gene-wise coefficient of variation
    ax = axes[0, 1]
    cv = (sample_replicates.std(axis=0) / (np.abs(sample_replicates.mean(axis=0)) + 1e-8)) * 100
    cv = np.clip(cv, 0, np.percentile(cv, 99))  # Cap at 99th percentile for visualization
    
    ax.hist(cv, bins=50, alpha=0.6, color=COLORS['primary'], 
           edgecolor='black', linewidth=0.5)
    ax.axvline(cv.mean(), color=COLORS['warning'], linestyle='--', 
              linewidth=2, label=f'Mean = {cv.mean():.1f}%')
    ax.axvline(np.median(cv), color=COLORS['success'], linestyle='--', 
              linewidth=2, label=f'Median = {np.median(cv):.1f}%')
    
    ax.set_xlabel('Coefficient of Variation (%)', fontsize=8)
    ax.set_ylabel('Number of Genes', fontsize=8)
    ax.set_title('Replicate Variability', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, frameon=False)
    sns.despine(ax=ax)
    
    # Panel C: Mean-variance relationship
    ax = axes[1, 0]
    gene_means = sample_replicates.mean(axis=0)
    gene_vars = sample_replicates.var(axis=0)
    
    ax.scatter(gene_means, gene_vars, s=5, alpha=0.4, 
              color=COLORS['primary'], edgecolors='none')
    
    ax.set_xlabel('Mean Expression', fontsize=8)
    ax.set_ylabel('Variance', fontsize=8)
    ax.set_title('Mean-Variance Relationship', fontsize=9, fontweight='bold')
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    sns.despine(ax=ax)
    
    # Panel D: Replicate dendrogram
    ax = axes[1, 1]
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform
    
    # Compute linkage
    dist_matrix = 1 - corr_matrix
    np.fill_diagonal(dist_matrix, 0)  # Ensure diagonal is 0
    condensed_dist = squareform(dist_matrix, checks=False)
    
    linkage_matrix = linkage(condensed_dist, method='average')
    dendrogram(linkage_matrix, ax=ax, labels=[f'Rep{i+1}' for i in range(n_replicates)],
              color_threshold=0.3, above_threshold_color='gray')
    
    ax.set_xlabel('Replicate', fontsize=8)
    ax.set_ylabel('Distance (1 - Correlation)', fontsize=8)
    ax.set_title('Hierarchical Clustering', fontsize=9, fontweight='bold')
    sns.despine(ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Replicate consistency plot saved to: {output_path}")
    return fig


def plot_heatmap_concordance(
    results_df: pd.DataFrame,
    group_by: List[str] = ['cell', 'dose'],
    metric: str = 'bai_score',
    top_n: int = 50,
    output_path: str = 'concordance_heatmap.png'
):
    """
    Create heatmap of concordance scores across conditions.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        BAI analysis results
    group_by : list
        Columns to use for heatmap axes
    metric : str
        Metric to display ('bai_score', 'mean_agreement', etc.)
    top_n : int
        Number of top compounds to display
    output_path : str
        Output file path
    """
    if len(group_by) != 2:
        raise ValueError("group_by must contain exactly 2 columns for heatmap axes")
    
    # Get top N compounds by BAI score
    top_compounds = results_df.nlargest(top_n, 'bai_score')
    
    # Create pivot table
    pivot_data = top_compounds.pivot_table(
        values=metric,
        index=group_by[0],
        columns=group_by[1],
        aggfunc='mean'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Heatmap
    im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', 
                  vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_xticklabels([str(x) for x in pivot_data.columns], fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels([str(x) for x in pivot_data.index], fontsize=7)
    
    ax.set_xlabel(group_by[1].replace('_', ' ').title(), fontsize=9)
    ax.set_ylabel(group_by[0].replace('_', ' ').title(), fontsize=9)
    ax.set_title(f'{metric.replace("_", " ").title()} Heatmap\n(Top {top_n} Compounds)', 
                fontsize=10, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric.replace('_', ' ').title(), fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    # Add values to cells
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            if not np.isnan(pivot_data.values[i, j]):
                text = ax.text(j, i, f'{pivot_data.values[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Concordance heatmap saved to: {output_path}")
    return fig
