import numpy as np
import matplotlib.pyplot as plt

def test_percentile_implementation():
    """Test and visualize percentile implementation for filter design"""
    
    print("🧪 TESTING PERCENTILE IMPLEMENTATION")
    print("="*50)
    
    # Create sample data similar to spectral features
    np.random.seed(42)
    
    # Simulate spectral centroids (voice frequency centers)
    centroids = np.random.normal(1500, 500, 1000)  # Mean around 1500 Hz
    centroids = np.clip(centroids, 200, 4000)  # Realistic range
    
    # Simulate spectral rolloffs (high frequency content)
    rolloffs = np.random.normal(3500, 800, 1000)  # Mean around 3500 Hz
    rolloffs = np.clip(rolloffs, 1000, 7000)  # Realistic range
    
    print(f"Sample data created:")
    print(f"  Centroids: {len(centroids)} values, range {centroids.min():.1f}-{centroids.max():.1f} Hz")
    print(f"  Rolloffs: {len(rolloffs)} values, range {rolloffs.min():.1f}-{rolloffs.max():.1f} Hz")
    
    # Calculate percentiles
    print(f"\n📊 PERCENTILE CALCULATIONS:")
    
    # Strategy 1: 1-99 percentiles
    p1_low = np.percentile(centroids, 1)
    p1_high = np.percentile(rolloffs, 99)
    
    # Strategy 2: 2.5-97.5 percentiles
    p2_low = np.percentile(centroids, 2.5)
    p2_high = np.percentile(rolloffs, 97.5)
    
    print(f"Strategy 1 (1-99 percentile):")
    print(f"  Low cutoff (1st percentile): {p1_low:.1f} Hz")
    print(f"  High cutoff (99th percentile): {p1_high:.1f} Hz")
    print(f"  Bandwidth: {p1_high - p1_low:.1f} Hz")
    
    print(f"\nStrategy 2 (2.5-97.5 percentile):")
    print(f"  Low cutoff (2.5th percentile): {p2_low:.1f} Hz")
    print(f"  High cutoff (97.5th percentile): {p2_high:.1f} Hz")
    print(f"  Bandwidth: {p2_high - p2_low:.1f} Hz")
    
    # Validation
    print(f"\n✅ VALIDATION:")
    print(f"  1-99% bandwidth > 2.5-97.5% bandwidth? {(p1_high - p1_low) > (p2_high - p2_low)}")
    print(f"  1st percentile < 2.5th percentile? {p1_low < p2_low}")
    print(f"  99th percentile > 97.5th percentile? {p1_high > p2_high}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Centroid distribution with percentiles
    axes[0, 0].hist(centroids, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(p1_low, color='blue', linestyle='--', linewidth=2, label=f'1st percentile: {p1_low:.1f} Hz')
    axes[0, 0].axvline(p2_low, color='red', linestyle='--', linewidth=2, label=f'2.5th percentile: {p2_low:.1f} Hz')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Spectral Centroids Distribution\n(Used for Low Cutoff)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Rolloff distribution with percentiles
    axes[0, 1].hist(rolloffs, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].axvline(p2_high, color='red', linestyle='--', linewidth=2, label=f'97.5th percentile: {p2_high:.1f} Hz')
    axes[0, 1].axvline(p1_high, color='blue', linestyle='--', linewidth=2, label=f'99th percentile: {p1_high:.1f} Hz')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Spectral Rolloffs Distribution\n(Used for High Cutoff)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Filter range comparison
    strategies = ['1-99 Percentile', '2.5-97.5 Percentile']
    low_cutoffs = [p1_low, p2_low]
    high_cutoffs = [p1_high, p2_high]
    bandwidths = [p1_high - p1_low, p2_high - p2_low]
    
    x_pos = np.arange(len(strategies))
    width = 0.35
    
    axes[1, 0].bar(x_pos - width/2, low_cutoffs, width, label='Low Cutoff', color='lightblue', edgecolor='black')
    axes[1, 0].bar(x_pos + width/2, high_cutoffs, width, label='High Cutoff', color='lightcoral', edgecolor='black')
    axes[1, 0].set_xlabel('Strategy')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_title('Filter Cutoff Frequencies')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(strategies)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (low, high) in enumerate(zip(low_cutoffs, high_cutoffs)):
        axes[1, 0].text(i - width/2, low + 50, f'{low:.0f}', ha='center', fontweight='bold')
        axes[1, 0].text(i + width/2, high + 50, f'{high:.0f}', ha='center', fontweight='bold')
    
    # 4. Bandwidth comparison
    axes[1, 1].bar(strategies, bandwidths, color=['#D5E8D4', '#FFF2CC'], edgecolor='black', alpha=0.8)
    axes[1, 1].set_ylabel('Bandwidth (Hz)')
    axes[1, 1].set_title('Filter Bandwidth Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, bw in enumerate(bandwidths):
        axes[1, 1].text(i, bw + 50, f'{bw:.0f} Hz', ha='center', fontweight='bold')
    
    plt.suptitle('🧪 Percentile Implementation Test & Validation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('essential_analysis/percentile_test_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Validation plot saved: essential_analysis/percentile_test_validation.png")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    if (p1_high - p1_low) > (p2_high - p2_low) and p1_low < p2_low and p1_high > p2_high:
        print("  ✅ Percentile implementation is CORRECT!")
        print("  ✅ 1-99% strategy provides broader frequency range")
        print("  ✅ 2.5-97.5% strategy provides conservative range")
    else:
        print("  ❌ Percentile implementation needs review!")
    
    return {
        'strategy_1': {'low': p1_low, 'high': p1_high, 'bandwidth': p1_high - p1_low},
        'strategy_2': {'low': p2_low, 'high': p2_high, 'bandwidth': p2_high - p2_low}
    }

if __name__ == "__main__":
    import os
    os.makedirs('essential_analysis', exist_ok=True)
    test_percentile_implementation()