"""
Analysis Script for Alpha Investigation Results
===============================================

Automatically analyzes results from investigate_alpha_grid.py:
1. Loads all result files
2. Creates visualizations
3. Generates summary report
4. Provides recommendations

Author: Phuong Nguyen
Date: 2024-11-06
"""

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AlphaResultsAnalyzer:
    """Analyzes and visualizes alpha investigation results."""
    
    def __init__(self, results_dir: str):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Directory containing investigation results
        """
        self.results_dir = Path(results_dir)
        
        if not self.results_dir.exists():
            raise ValueError(f"Results directory not found: {results_dir}")
        
        self.summary_df = None
        self.all_results = []
        
        print(f"Analyzing results from: {self.results_dir}")
    
    def load_results(self):
        """Load all result JSON files and summary CSV."""
        print("\n" + "="*60)
        print("Loading results...")
        
        # Load summary CSV
        summary_file = self.results_dir / "alpha_investigation_summary.csv"
        if summary_file.exists():
            self.summary_df = pd.read_csv(summary_file)
            print(f"✓ Loaded summary with {len(self.summary_df)} experiments")
        else:
            print("⚠ Summary CSV not found")
        
        # Load individual result JSONs
        result_files = sorted(self.results_dir.glob("alpha_*.json"))
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    self.all_results.append(result)
            except Exception as e:
                print(f"⚠ Error loading {result_file.name}: {e}")
        
        print(f"✓ Loaded {len(self.all_results)} detailed results")
        print("="*60)
    
    def create_visualizations(self):
        """Create comprehensive visualization suite."""
        print("\nCreating visualizations...")
        
        fig_dir = self.results_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        # 1. Main trade-off plot: Alpha vs C-index and Sparsity
        self._plot_alpha_tradeoff(fig_dir)
        
        # 2. Sparsity metrics comparison
        self._plot_sparsity_metrics(fig_dir)
        
        # 3. Training curves for each alpha
        self._plot_training_curves(fig_dir)
        
        # 4. Feature importance distributions
        self._plot_importance_distributions(fig_dir)
        
        # 5. Gradient stability analysis
        self._plot_gradient_analysis(fig_dir)
        
        print(f"✓ Figures saved to: {fig_dir}")
    
    def _plot_alpha_tradeoff(self, fig_dir: Path):
        """Plot the fundamental alpha-performance-sparsity trade-off."""
        if self.summary_df is None:
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Sort by alpha for cleaner lines
        df = self.summary_df.sort_values('alpha')
        
        # Plot 1: Alpha vs C-index
        ax1.plot(df['alpha'], df['final_c_index'], 'o-', linewidth=2, markersize=8)
        ax1.axhline(y=0.72, color='r', linestyle='--', label='Chapter 2 (TCGA): 0.72', alpha=0.7)
        ax1.axhline(y=0.68, color='orange', linestyle='--', label='Chapter 2 (ORIEN): 0.68', alpha=0.7)
        ax1.set_xlabel('Alpha (Regularization Strength)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Validation C-index', fontsize=12, fontweight='bold')
        ax1.set_title('Performance vs Regularization', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Alpha vs Sparsity
        ax2.plot(df['alpha'], df['sparsity_1e-3'], 'o-', linewidth=2, markersize=8, 
                label='Sparsity (<1e-3)')
        ax2.plot(df['alpha'], df['sparsity_1e-4'], 's-', linewidth=2, markersize=8, 
                label='Sparsity (<1e-4)')
        ax2.set_xlabel('Alpha (Regularization Strength)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Sparsity (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Sparsity vs Regularization', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: C-index vs Sparsity (Pareto frontier)
        scatter = ax3.scatter(df['sparsity_1e-3'], df['final_c_index'], 
                            c=df['alpha'], s=200, cmap='viridis', alpha=0.7)
        
        # Annotate points with alpha values
        for idx, row in df.iterrows():
            ax3.annotate(f"α={row['alpha']:.3f}", 
                        (row['sparsity_1e-3'], row['final_c_index']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Sparsity (<1e-3) (%)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Validation C-index', fontsize=12, fontweight='bold')
        ax3.set_title('Performance-Sparsity Trade-off', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add Chapter 2 target zone
        ax3.axhspan(0.68, 0.72, alpha=0.2, color='green', label='Chapter 2 range')
        ax3.legend()
        
        # Add colorbar for alpha
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Alpha', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'alpha_tradeoff_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Created: alpha_tradeoff_analysis.png")
    
    def _plot_sparsity_metrics(self, fig_dir: Path):
        """Plot detailed sparsity metrics."""
        if self.summary_df is None:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        df = self.summary_df.sort_values('alpha')
        
        # Plot 1: Number of non-zero genes
        ax1.bar(range(len(df)), df['num_nonzero'], alpha=0.7)
        ax1.axhline(y=20, color='r', linestyle='--', label='Chapter 2: 20 genes', alpha=0.7)
        ax1.axhline(y=50, color='orange', linestyle='--', label='Target: <50 genes', alpha=0.7)
        ax1.set_xlabel('Alpha Value', fontsize=11)
        ax1.set_ylabel('Number of Non-Zero Genes', fontsize=11)
        ax1.set_title('Effective Gene Count', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels([f"{a:.3f}" for a in df['alpha']], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Importance ratio (max/median)
        ax2.plot(df['alpha'], df['importance_ratio'], 'o-', linewidth=2, markersize=8, color='purple')
        ax2.axhline(y=50, color='g', linestyle='--', label='Target: >50x', alpha=0.7)
        ax2.set_xlabel('Alpha (log scale)', fontsize=11)
        ax2.set_ylabel('Importance Ratio (max/median)', fontsize=11)
        ax2.set_title('Feature Importance Concentration', fontsize=12, fontweight='bold')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Top-10 gene concentration
        ax3.plot(df['alpha'], df['top_10_concentration'], 'o-', linewidth=2, markersize=8, color='teal')
        ax3.set_xlabel('Alpha (log scale)', fontsize=11)
        ax3.set_ylabel('Top-10 Importance (%)', fontsize=11)
        ax3.set_title('Concentration in Top 10 Genes', fontsize=12, fontweight='bold')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Gradient norms
        ax4.plot(df['alpha'], df['mean_grad_norm'], 'o-', linewidth=2, markersize=8, 
                label='Mean', color='blue')
        ax4.plot(df['alpha'], df['max_grad_norm'], 's-', linewidth=2, markersize=8, 
                label='Max', color='red')
        ax4.axhline(y=1.0, color='g', linestyle='--', label='Stable threshold', alpha=0.7)
        ax4.set_xlabel('Alpha (log scale)', fontsize=11)
        ax4.set_ylabel('Gradient Norm', fontsize=11)
        ax4.set_title('Gradient Stability', fontsize=12, fontweight='bold')
        ax4.set_xscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'sparsity_metrics_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Created: sparsity_metrics_detailed.png")
    
    def _plot_training_curves(self, fig_dir: Path):
        """Plot training curves for each alpha value."""
        if not self.all_results:
            return
        
        n_alphas = len(self.all_results)
        n_cols = 3
        n_rows = (n_alphas + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_alphas > 1 else [axes]
        
        for idx, result in enumerate(sorted(self.all_results, key=lambda x: x['alpha'])):
            alpha = result['alpha']
            curves = result['training_curves']
            
            ax = axes[idx]
            
            # Plot validation C-index
            ax.plot(curves['val_c_index'], linewidth=2, label='Val C-index', color='blue')
            ax.axhline(y=0.72, color='r', linestyle='--', alpha=0.5, label='Ch2: 0.72')
            ax.axhline(y=0.68, color='orange', linestyle='--', alpha=0.5, label='Ch2: 0.68')
            
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('C-index', fontsize=10)
            ax.set_title(f'Alpha = {alpha:.4f}\nFinal: {result["final_val_c_index"]:.4f}',
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(self.all_results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'training_curves_all_alphas.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Created: training_curves_all_alphas.png")
    
    def _plot_importance_distributions(self, fig_dir: Path):
        """Plot feature importance distributions for each alpha."""
        if not self.all_results:
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, result in enumerate(sorted(self.all_results, key=lambda x: x['alpha'])):
            alpha = result['alpha']
            importances = np.array(result['feature_importances'])
            
            ax = axes[idx]
            
            # Histogram of importance values (log scale)
            ax.hist(np.log10(importances + 1e-10), bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(x=np.log10(1e-4), color='r', linestyle='--', 
                      label='Threshold: 1e-4', linewidth=2)
            ax.axvline(x=np.log10(1e-3), color='orange', linestyle='--', 
                      label='Threshold: 1e-3', linewidth=2)
            
            ax.set_xlabel('Log10(Importance)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Alpha = {alpha:.4f}\nSparsity: {result["sparsity_metrics"]["sparsity_1e-3"]:.1f}%',
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'importance_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Created: importance_distributions.png")
    
    def _plot_gradient_analysis(self, fig_dir: Path):
        """Analyze gradient behavior across alphas."""
        if not self.all_results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for result in sorted(self.all_results, key=lambda x: x['alpha']):
            alpha = result['alpha']
            grad_norms = result['training_curves']['gradient_norms']
            epochs = range(len(grad_norms))
            
            # Plot gradient evolution
            ax1.plot(epochs, grad_norms, alpha=0.7, label=f'α={alpha:.3f}')
        
        ax1.axhline(y=1.0, color='k', linestyle='--', linewidth=2, label='Stable threshold')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Gradient Norm', fontsize=11)
        ax1.set_title('Gradient Evolution During Training', fontsize=12, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Summary statistics
        alphas = [r['alpha'] for r in self.all_results]
        mean_grads = [r['gradient_statistics']['mean'] for r in self.all_results]
        max_grads = [r['gradient_statistics']['max'] for r in self.all_results]
        
        ax2.plot(alphas, mean_grads, 'o-', linewidth=2, markersize=8, label='Mean', color='blue')
        ax2.plot(alphas, max_grads, 's-', linewidth=2, markersize=8, label='Max', color='red')
        ax2.axhline(y=1.0, color='g', linestyle='--', linewidth=2, label='Stable threshold')
        ax2.set_xlabel('Alpha (log scale)', fontsize=11)
        ax2.set_ylabel('Gradient Norm', fontsize=11)
        ax2.set_title('Gradient Statistics vs Alpha', fontsize=12, fontweight='bold')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'gradient_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Created: gradient_analysis.png")
    
    def generate_report(self):
        """Generate text summary report with recommendations."""
        print("\n" + "="*60)
        print("ANALYSIS REPORT")
        print("="*60)
        
        if self.summary_df is None:
            print("⚠ No summary data available")
            return
        
        report_lines = []
        report_lines.append("\n📊 INVESTIGATION SUMMARY\n")
        report_lines.append(f"Total experiments: {len(self.summary_df)}")
        report_lines.append(f"Alpha range tested: {self.summary_df['alpha'].min():.4f} to {self.summary_df['alpha'].max():.4f}")
        report_lines.append(f"Dataset: TCGA (n={339}, genes={308})")
        
        # Performance analysis
        report_lines.append("\n🎯 PERFORMANCE ANALYSIS\n")
        best_c_idx = self.summary_df.loc[self.summary_df['final_c_index'].idxmax()]
        report_lines.append(f"Best C-index: {best_c_idx['final_c_index']:.4f} (alpha={best_c_idx['alpha']:.4f})")
        report_lines.append(f"Chapter 2 baseline: 0.72 (TCGA), 0.68 (ORIEN)")
        
        perf_gap = 0.68 - best_c_idx['final_c_index']
        report_lines.append(f"Performance gap: {perf_gap:.4f} ({100*perf_gap/0.68:.1f}%)")
        
        # Sparsity analysis
        report_lines.append("\n🔍 SPARSITY ANALYSIS\n")
        best_sparse_idx = self.summary_df.loc[self.summary_df['sparsity_1e-3'].idxmax()]
        report_lines.append(f"Maximum sparsity: {best_sparse_idx['sparsity_1e-3']:.1f}% (alpha={best_sparse_idx['alpha']:.4f})")
        report_lines.append(f"At best C-index: {best_c_idx['sparsity_1e-3']:.1f}% sparsity")
        report_lines.append(f"At max sparsity: {best_sparse_idx['final_c_index']:.4f} C-index")
        
        # Trade-off analysis
        report_lines.append("\n⚖️ TRADE-OFF ANALYSIS\n")
        
        # Find candidates that meet criteria
        candidates = self.summary_df[
            (self.summary_df['final_c_index'] >= 0.68) & 
            (self.summary_df['num_nonzero'] <= 50)
        ]
        
        if len(candidates) > 0:
            report_lines.append(f"✅ Found {len(candidates)} configurations meeting targets:")
            report_lines.append("   - C-index ≥ 0.68")
            report_lines.append("   - ≤ 50 non-zero genes")
            report_lines.append("\nBest candidates:")
            for _, row in candidates.iterrows():
                report_lines.append(
                    f"  α={row['alpha']:.4f}: C-index={row['final_c_index']:.4f}, "
                    f"genes={row['num_nonzero']:.0f}, sparsity={row['sparsity_1e-3']:.1f}%"
                )
        else:
            report_lines.append("❌ No configuration meets both targets:")
            report_lines.append("   - C-index ≥ 0.68 AND ≤ 50 non-zero genes")
            
            # Find closest compromises
            report_lines.append("\n📍 Closest compromises:")
            
            # Best C-index with reasonable sparsity
            reasonable = self.summary_df[self.summary_df['sparsity_1e-3'] >= 20]
            if len(reasonable) > 0:
                best_reasonable = reasonable.loc[reasonable['final_c_index'].idxmax()]
                report_lines.append(
                    f"  Best C-index (≥20% sparse): α={best_reasonable['alpha']:.4f}, "
                    f"C-index={best_reasonable['final_c_index']:.4f}, "
                    f"genes={best_reasonable['num_nonzero']:.0f}"
                )
            
            # Best sparsity with reasonable C-index
            reasonable2 = self.summary_df[self.summary_df['final_c_index'] >= 0.60]
            if len(reasonable2) > 0:
                best_sparse = reasonable2.loc[reasonable2['sparsity_1e-3'].idxmax()]
                report_lines.append(
                    f"  Best sparsity (C-index ≥0.60): α={best_sparse['alpha']:.4f}, "
                    f"C-index={best_sparse['final_c_index']:.4f}, "
                    f"sparsity={best_sparse['sparsity_1e-3']:.1f}%"
                )
        
        # Root cause confirmation
        report_lines.append("\n🔬 ROOT CAUSE CONFIRMATION\n")
        
        weak_alpha = self.summary_df[self.summary_df['alpha'] <= 0.01]
        strong_alpha = self.summary_df[self.summary_df['alpha'] >= 0.1]
        
        if len(weak_alpha) > 0 and len(strong_alpha) > 0:
            weak_sparsity = weak_alpha['sparsity_1e-3'].mean()
            strong_sparsity = strong_alpha['sparsity_1e-3'].mean()
            sparsity_increase = strong_sparsity - weak_sparsity
            
            report_lines.append(f"Weak alpha (≤0.01): {weak_sparsity:.1f}% average sparsity")
            report_lines.append(f"Strong alpha (≥0.1): {strong_sparsity:.1f}% average sparsity")
            report_lines.append(f"Sparsity increase: +{sparsity_increase:.1f}%")
            
            if sparsity_increase > 50:
                report_lines.append("✅ CONFIRMED: Alpha is root cause of sparsity failure")
            else:
                report_lines.append("⚠️ PARTIAL: Alpha has moderate effect on sparsity")
        
        # Recommendations
        report_lines.append("\n💡 RECOMMENDATIONS\n")
        
        if len(candidates) > 0:
            best_candidate = candidates.loc[candidates['final_c_index'].idxmax()]
            report_lines.append(f"1. Use alpha = {best_candidate['alpha']:.4f} for Phase 2")
            report_lines.append("2. Test bidirectional validation (TCGA ↔ ORIEN)")
            report_lines.append("3. Compare biomarker overlap with Chapter 2")
        else:
            # Provide strategic options
            report_lines.append("Based on results, consider these options:")
            report_lines.append("\n Option A: Accept lower C-index for sparsity")
            if len(reasonable2) > 0:
                best = reasonable2.loc[reasonable2['sparsity_1e-3'].idxmax()]
                report_lines.append(f"  → Use alpha = {best['alpha']:.4f}")
                report_lines.append(f"  → Expected: C-index ~{best['final_c_index']:.2f}, sparse biomarkers")
            
            report_lines.append("\n Option B: Hybrid approach")
            report_lines.append("  → Cox elastic net pre-filtering (Chapter 2 method)")
            report_lines.append("  → DeepSurv refinement on selected genes")
            
            report_lines.append("\n Option C: Document limitation")
            report_lines.append("  → Valid scientific finding: DL requires larger samples")
            report_lines.append("  → Compare to literature (Haibe-Kains et al. 2013)")
        
        # Next steps
        report_lines.append("\n📋 NEXT STEPS\n")
        report_lines.append("1. Review visualizations in figures/ directory")
        report_lines.append("2. Discuss results with advisor")
        report_lines.append("3. Decide on strategy: A, B, or C")
        if len(candidates) > 0 or len(reasonable2) > 0:
            report_lines.append("4. Proceed to Phase 2: bidirectional validation")
        
        report = "\n".join(report_lines)
        print(report)
        
        # Save report
        report_file = self.results_dir / "analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to: {report_file}")
        print("="*60)
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        self.load_results()
        self.create_visualizations()
        self.generate_report()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze alpha investigation results')
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing investigation results'
    )
    
    args = parser.parse_args()
    
    analyzer = AlphaResultsAnalyzer(args.results_dir)
    analyzer.run_analysis()
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
