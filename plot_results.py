"""
Visualization script for grokking training results.

Usage:
    python plot_results.py runs/experiment/training_history.json
    python plot_results.py runs/experiment/training_history.json --output grokking_plot.png
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_training_history(json_path: str) -> dict:
    """Load training history from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_grokking_curves(history: dict, output_path: str = None, show: bool = True):
    """
    Create a comprehensive visualization of the grokking phenomenon.

    Args:
        history: Dictionary containing training metrics
        output_path: Path to save the plot (optional)
        show: Whether to display the plot interactively
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Grokking Phenomenon: Delayed Generalization', fontsize=16, fontweight='bold')

    epochs = history['epoch']

    # Plot 1: Accuracy Comparison (Main Grokking Plot)
    ax1 = axes[0, 0]
    ax1.plot(epochs, [acc * 100 for acc in history['train_acc']],
             label='Train Accuracy', linewidth=2, color='#2E86AB', marker='o', markersize=3)
    ax1.plot(epochs, [acc * 100 for acc in history['val_acc']],
             label='Validation Accuracy', linewidth=2, color='#A23B72', marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy: The Grokking Transition', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Add annotation for grokking transition
    # Find approximate grokking point (when val_acc increases significantly)
    val_accs = history['val_acc']
    for i in range(1, len(val_accs)):
        if val_accs[i] > 0.5 and val_accs[i-1] < 0.5:
            grok_epoch = epochs[i]
            ax1.axvline(x=grok_epoch, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax1.annotate('Grokking!', xy=(grok_epoch, 50), xytext=(grok_epoch + 10, 30),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=11, color='red', fontweight='bold')
            break

    # Plot 2: Loss Comparison
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_loss'],
             label='Train Loss', linewidth=2, color='#2E86AB', marker='o', markersize=3)
    ax2.plot(epochs, history['val_loss'],
             label='Validation Loss', linewidth=2, color='#A23B72', marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Curves', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: Generalization Gap
    ax3 = axes[1, 0]
    gen_gap = [(train - val) * 100 for train, val in zip(history['train_acc'], history['val_acc'])]
    ax3.plot(epochs, gen_gap, linewidth=2.5, color='#F18F01', marker='D', markersize=4)
    ax3.fill_between(epochs, 0, gen_gap, alpha=0.3, color='#F18F01')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Generalization Gap (%)', fontsize=12)
    ax3.set_title('Memorization → Generalization', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learning Rate and Weight Norm
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    line1 = ax4.plot(epochs, history['lr'],
                     label='Learning Rate', linewidth=2, color='#06A77D', marker='^', markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12, color='#06A77D')
    ax4.tick_params(axis='y', labelcolor='#06A77D')

    line2 = ax4_twin.plot(epochs, history['weight_norm'],
                          label='Weight Norm', linewidth=2, color='#D62246', marker='v', markersize=4)
    ax4_twin.set_ylabel('Weight Norm (L2)', fontsize=12, color='#D62246')
    ax4_twin.tick_params(axis='y', labelcolor='#D62246')

    ax4.set_title('Training Dynamics', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=10)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Add footer with metrics
    final_train_acc = history['train_acc'][-1] * 100
    final_val_acc = history['val_acc'][-1] * 100
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]

    footer_text = (f"Final Metrics: Train Acc={final_train_acc:.2f}% | Val Acc={final_val_acc:.2f}% | "
                  f"Train Loss={final_train_loss:.4f} | Val Loss={final_val_loss:.4f}")
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Save plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    # Show plot
    if show:
        plt.show()

    plt.close()


def create_simple_grokking_plot(history: dict, output_path: str = None, show: bool = True):
    """
    Create a simple, clean grokking plot suitable for README.

    Args:
        history: Dictionary containing training metrics
        output_path: Path to save the plot (optional)
        show: Whether to display the plot interactively
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = history['epoch']

    # Plot accuracy curves
    train_line = ax.plot(epochs, [acc * 100 for acc in history['train_acc']],
                         label='Training Accuracy', linewidth=3, color='#2E86AB', marker='o', markersize=5)
    val_line = ax.plot(epochs, [acc * 100 for acc in history['val_acc']],
                       label='Validation Accuracy', linewidth=3, color='#A23B72', marker='s', markersize=5)

    # Add grokking annotation
    val_accs = history['val_acc']
    for i in range(1, len(val_accs)):
        if val_accs[i] > 0.5 and val_accs[i-1] < 0.5:
            grok_epoch = epochs[i]
            ax.axvline(x=grok_epoch, color='red', linestyle='--', alpha=0.6, linewidth=2.5)
            ax.annotate('Grokking Transition', xy=(grok_epoch, 50), xytext=(grok_epoch + 15, 35),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
                       fontsize=14, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

            # Add shaded regions
            ax.axvspan(0, grok_epoch, alpha=0.1, color='red', label='Memorization Phase')
            ax.axvspan(grok_epoch, epochs[-1], alpha=0.1, color='green', label='Generalization Phase')
            break

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Grokking: Sudden Delayed Generalization on Modular Arithmetic',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])

    # Add final metrics
    final_train_acc = history['train_acc'][-1] * 100
    final_val_acc = history['val_acc'][-1] * 100
    metrics_text = f"Final: Train={final_train_acc:.1f}%, Val={final_val_acc:.1f}%"
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Simple plot saved to: {output_path}")

    if show:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize grokking training results')
    parser.add_argument('json_path', type=str, help='Path to training_history.json file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for the plot (default: same directory as JSON)')
    parser.add_argument('--simple', action='store_true',
                       help='Create simple plot suitable for README')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the plot interactively')

    args = parser.parse_args()

    # Load training history
    history = load_training_history(args.json_path)

    # Determine output path
    if args.output is None:
        json_dir = Path(args.json_path).parent
        filename = 'grokking_plot_simple.png' if args.simple else 'grokking_plot.png'
        output_path = json_dir / filename
    else:
        output_path = args.output

    # Create plot
    show = not args.no_show
    if args.simple:
        create_simple_grokking_plot(history, str(output_path), show=show)
    else:
        plot_grokking_curves(history, str(output_path), show=show)

    print(f"\nTraining Summary:")
    print(f"  Total epochs: {len(history['epoch'])}")
    print(f"  Final train accuracy: {history['train_acc'][-1] * 100:.2f}%")
    print(f"  Final validation accuracy: {history['val_acc'][-1] * 100:.2f}%")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final validation loss: {history['val_loss'][-1]:.4f}")

    # Detect grokking point
    val_accs = history['val_acc']
    for i in range(1, len(val_accs)):
        if val_accs[i] > 0.5 and val_accs[i-1] < 0.5:
            print(f"  Grokking transition: ~Epoch {history['epoch'][i]}")
            break


if __name__ == '__main__':
    main()
