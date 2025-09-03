import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

def create_upgraded_processing_pipeline_diagram():
    """Create an upgraded processing pipeline diagram showing percentile-based filtering"""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'analysis': '#FFF2CC', 
        'percentile': '#D5E8D4',
        'filtering': '#F8CECC',
        'output': '#E1D5E7',
        'arrow': '#666666'
    }
    
    # Helper function to create rounded rectangles
    def create_box(xy, width, height, text, color, text_color='black', fontsize=10):
        box = FancyBboxPatch(xy, width, height,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(xy[0] + width/2, xy[1] + height/2, text,
               ha='center', va='center', fontsize=fontsize, fontweight='bold',
               color=text_color, wrap=True)
    
    # Helper function to create arrows
    def create_arrow(start, end, text='', offset=0.1):
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors['arrow']))
        if text:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2 + offset
            ax.text(mid_x, mid_y, text, ha='center', va='center', 
                   fontsize=8, style='italic', color=colors['arrow'])
    
    # Title
    ax.text(10, 13, 'Advanced Audio Preprocessing Pipeline for Parkinson\'s Disease Analysis',
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(10, 12.5, 'Percentile-Based Band-Pass Filtering with Multiple Strategies',
            ha='center', va='center', fontsize=12, style='italic', color='darkblue')
    
    # 1. Input Data
    create_box((1, 11), 3, 0.8, 'Raw Audio Data\n(PD: 2 files, HC: 19 files)\n16kHz Sampling', 
               colors['input'], fontsize=9)
    
    # 2. Dataset Analysis
    create_box((6, 11), 3.5, 0.8, 'Dataset Analysis\n• Audio Length Distribution\n• Silence Ratio Analysis\n• Frequency Profile Analysis', 
               colors['analysis'], fontsize=9)
    
    # 3. Frequency Analysis
    create_box((11, 11), 4, 0.8, 'Frequency Percentile Calculation\n• Spectral Centroid Analysis\n• Spectral Rolloff Analysis\n• Energy Distribution', 
               colors['analysis'], fontsize=9)
    
    # 4. Percentile Strategies (Three parallel paths)
    # Strategy 1: 1st-99th Percentile
    create_box((1, 9), 2.8, 1, '1st-99th Percentile\nStrategy\n380.9-1257.8 Hz\n(Broadest Range)', 
               colors['percentile'], fontsize=8)
    
    # Strategy 2: 2.5th-97.5th Percentile  
    create_box((5, 9), 2.8, 1, '2.5th-97.5th Percentile\nStrategy\n422.1-1234.4 Hz\n(Conservative)', 
               colors['percentile'], fontsize=8)
    
    # Strategy 3: 95% Energy Range
    create_box((9, 9), 2.8, 1, '95% Energy Range\nStrategy\n125.0-1789.1 Hz\n(Energy-Based)', 
               colors['percentile'], fontsize=8)
    
    # 5. Band-Pass Filtering (Three parallel paths)
    create_box((1, 7), 2.8, 1, 'Band-Pass Filter 1\n3rd Order Butterworth\n380.9-1257.8 Hz\nPreserves 99% content', 
               colors['filtering'], fontsize=8)
    
    create_box((5, 7), 2.8, 1, 'Band-Pass Filter 2\n3rd Order Butterworth\n422.1-1234.4 Hz\nConservative filtering', 
               colors['filtering'], fontsize=8)
    
    create_box((9, 7), 2.8, 1, 'Band-Pass Filter 3\n3rd Order Butterworth\n125.0-1789.1 Hz\nEnergy-optimized', 
               colors['filtering'], fontsize=8)
    
    # 6. Silence Analysis & Removal
    create_box((13, 8), 3, 1, 'Silence Analysis\n• Frame-based Energy\n• Percentile Thresholding\n• Selective Removal', 
               colors['analysis'], fontsize=9)
    
    # 7. Final Processing
    create_box((6, 5), 4, 0.8, 'Final Processing\n• NO Amplitude Normalization\n• NO Length Standardization\n• Preserve Original Characteristics', 
               colors['filtering'], fontsize=9)
    
    # 8. Output Data (Three parallel outputs)
    create_box((1, 3), 2.8, 1, 'Output 1\npercentile_1_99_\nfiltered_data/\n(Primary Output)', 
               colors['output'], fontsize=8)
    
    create_box((5, 3), 2.8, 1, 'Output 2\npercentile_2_5_97_5_\nfiltered_data/\n(Conservative)', 
               colors['output'], fontsize=8)
    
    create_box((9, 3), 2.8, 1, 'Output 3\nfrequency_range_95_\nfiltered_data/\n(Energy-Based)', 
               colors['output'], fontsize=8)
    
    # 9. Analysis Outputs
    create_box((13, 5), 3, 1.5, 'Analysis Outputs\n• Audio Length Distribution\n• Silence Analysis\n• Frequency Profiles\n• Below 80Hz Analysis\n• Signal Statistics', 
               colors['output'], fontsize=9)
    
    # 10. Feature Extraction (Next Step)
    create_box((6, 1), 4, 0.8, 'Next Step: Feature Extraction\n• Time Domain Features\n• Frequency Domain Features\n• MFCC Features', 
               '#FFE6CC', fontsize=9)
    
    # Add arrows connecting the components
    # Main flow
    create_arrow((4, 11.4), (6, 11.4))
    create_arrow((9.5, 11.4), (11, 11.4))
    
    # From analysis to strategies
    create_arrow((7.5, 11), (2.4, 10))
    create_arrow((7.5, 11), (6.4, 10))
    create_arrow((7.5, 11), (10.4, 10))
    
    # From strategies to filtering
    create_arrow((2.4, 9), (2.4, 8))
    create_arrow((6.4, 9), (6.4, 8))
    create_arrow((10.4, 9), (10.4, 8))
    
    # From filtering to silence analysis
    create_arrow((3.8, 7.5), (13, 8.5))
    create_arrow((7.8, 7.5), (13, 8.5))
    create_arrow((11.8, 7.5), (13, 8.5))
    
    # From silence analysis to final processing
    create_arrow((13, 8), (10, 5.4))
    
    # From final processing to outputs
    create_arrow((6.5, 5), (2.4, 4))
    create_arrow((8, 5), (6.4, 4))
    create_arrow((9.5, 5), (10.4, 4))
    
    # From outputs to feature extraction
    create_arrow((2.4, 3), (6.5, 1.8))
    create_arrow((6.4, 3), (7.5, 1.8))
    create_arrow((10.4, 3), (8.5, 1.8))
    
    # From analysis to analysis outputs
    create_arrow((14.5, 8), (14.5, 6.5))
    
    # Add legend
    legend_x = 16.5
    legend_y = 11
    ax.text(legend_x, legend_y, 'Legend:', fontsize=12, fontweight='bold')
    
    legend_items = [
        ('Input Data', colors['input']),
        ('Analysis', colors['analysis']),
        ('Percentile Strategy', colors['percentile']),
        ('Filtering', colors['filtering']),
        ('Output', colors['output'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y - 0.4 - (i * 0.3)
        rect = Rectangle((legend_x - 0.2, y_pos - 0.08), 0.3, 0.16, 
                        facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(legend_x + 0.2, y_pos, label, fontsize=10, va='center')
    
    # Add processing statistics
    stats_text = """Processing Statistics:
• Total Files: 21 (2 PD, 19 HC)
• Filter Strategies: 3 parallel approaches
• Length Retention: 100% (no truncation)
• Silence Handling: Selective removal
• Amplitude: Preserved (no normalization)
• Duration: Natural variability maintained"""
    
    ax.text(16.5, 3.5, stats_text, fontsize=9, va='top', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    # Add key improvements
    improvements_text = """Key Improvements:
✓ Percentile-based frequency analysis
✓ Multiple filtering strategies
✓ Dataset-specific parameter calculation
✓ Enhanced band-pass filtering
✓ Comprehensive visualization suite
✓ Parallel processing outputs"""
    
    ax.text(1, 0.5, improvements_text, fontsize=9, va='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    # Set axis properties
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save the diagram
    plt.tight_layout()
    plt.savefig('upgraded_processing_pipeline_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('preprocessing_analysis/upgraded_processing_pipeline_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Upgraded processing pipeline diagram created!")
    print("📁 Saved as: upgraded_processing_pipeline_diagram.png")
    print("📁 Also saved in: preprocessing_analysis/upgraded_processing_pipeline_diagram.png")

def create_frequency_filtering_comparison_diagram():
    """Create a diagram comparing the three filtering strategies"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Strategy comparison data
    strategies = {
        '1st-99th Percentile': {'range': (380.9, 1257.8), 'color': '#D5E8D4'},
        '2.5th-97.5th Percentile': {'range': (422.1, 1234.4), 'color': '#FFF2CC'},
        '95% Energy Range': {'range': (125.0, 1789.1), 'color': '#F8CECC'}
    }
    
    # 1. Frequency Range Comparison
    ax = axes[0, 0]
    y_pos = np.arange(len(strategies))
    
    for i, (strategy, data) in enumerate(strategies.items()):
        low, high = data['range']
        bandwidth = high - low
        ax.barh(i, bandwidth, left=low, color=data['color'], alpha=0.7, edgecolor='black')
        ax.text(low + bandwidth/2, i, f'{low:.1f}-{high:.1f} Hz\n({bandwidth:.1f} Hz BW)', 
               ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(strategies.keys())
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('Frequency Range Comparison', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 2. Bandwidth Comparison
    ax = axes[0, 1]
    bandwidths = [data['range'][1] - data['range'][0] for data in strategies.values()]
    colors = [data['color'] for data in strategies.values()]
    
    bars = ax.bar(range(len(strategies)), bandwidths, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([s.replace(' ', '\n') for s in strategies.keys()], fontsize=9)
    ax.set_ylabel('Bandwidth (Hz)')
    ax.set_title('Bandwidth Comparison', fontweight='bold', fontsize=12)
    
    # Add bandwidth values on bars
    for bar, bw in zip(bars, bandwidths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
               f'{bw:.1f} Hz', ha='center', va='bottom', fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    # 3. Coverage Analysis
    ax = axes[1, 0]
    coverage_data = {
        '1st-99th Percentile': 99,
        '2.5th-97.5th Percentile': 95,
        '95% Energy Range': 95
    }
    
    wedges, texts, autotexts = ax.pie(coverage_data.values(), 
                                     labels=coverage_data.keys(),
                                     autopct='%1.1f%%',
                                     colors=[data['color'] for data in strategies.values()],
                                     startangle=90)
    ax.set_title('Frequency Content Coverage', fontweight='bold', fontsize=12)
    
    # 4. Strategy Recommendations
    ax = axes[1, 1]
    ax.axis('off')
    
    recommendations = """Strategy Recommendations:

🎯 1st-99th Percentile (PRIMARY)
   • Best for: Maximum signal preservation
   • Use case: Primary analysis pipeline
   • Pros: Captures 99% of frequency content
   • Cons: May include some noise

🛡️ 2.5th-97.5th Percentile (CONSERVATIVE)
   • Best for: Noise-sensitive analysis
   • Use case: Conservative preprocessing
   • Pros: Better noise rejection
   • Cons: Slightly reduced content

⚡ 95% Energy Range (ENERGY-BASED)
   • Best for: Energy-focused analysis
   • Use case: Spectral energy studies
   • Pros: Widest frequency range
   • Cons: May include more noise

✅ Recommendation: Use 1st-99th percentile
   as primary strategy for PD analysis"""
    
    ax.text(0.05, 0.95, recommendations, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Percentile-Based Filtering Strategy Comparison', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('filtering_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('preprocessing_analysis/filtering_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Filtering strategy comparison diagram created!")
    print("📁 Saved as: filtering_strategy_comparison.png")

if __name__ == "__main__":
    print("🎨 Creating upgraded processing pipeline diagrams...")
    create_upgraded_processing_pipeline_diagram()
    create_frequency_filtering_comparison_diagram()
    print("\n✨ All diagrams created successfully!")
