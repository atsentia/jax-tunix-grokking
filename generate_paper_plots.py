"""Generate plots for the distilled grokking paper."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2


def load_history(run_dir):
    """Load training/distillation history from a run directory."""
    history_files = [
        'training_history.json',
        'distillation_history.json'
    ]

    for filename in history_files:
        path = Path(run_dir) / filename
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    raise FileNotFoundError(f"No history file found in {run_dir}")


def plot_comparison():
    """Create comparison plot of all three distillation strategies."""

    # Load data
    teacher_data = load_history('runs/teacher_adamw')
    logit_data = load_history('runs/distill_0.5x_logit')
    attention_data = load_history('runs/distill_0.5x_attention')
    feature_data = load_history('runs/distill_0.5x_feature')

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # === Left plot: Validation Accuracy ===
    ax1.plot(teacher_data['epoch'],
             [acc * 100 for acc in teacher_data['val_acc']],
             label='Teacher (128-dim)',
             linestyle='--',
             color='black',
             linewidth=2.5,
             alpha=0.7)

    ax1.plot(logit_data['epoch'],
             [acc * 100 for acc in logit_data['val_acc']],
             label='Logit Distillation (64-dim)',
             color='red',
             marker='o',
             markevery=5)

    ax1.plot(attention_data['epoch'],
             [acc * 100 for acc in attention_data['val_acc']],
             label='Attention Transfer (64-dim)',
             color='blue',
             marker='s',
             markevery=5)

    ax1.plot(feature_data['epoch'],
             [acc * 100 for acc in feature_data['val_acc']],
             label='Feature Projection (64-dim)',
             color='green',
             marker='^',
             markevery=5)

    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=14)
    ax1.set_title('Validation Accuracy: Distillation Strategies', fontsize=16, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)

    # Add annotation for teacher grokking transition
    ax1.axvline(x=70, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.text(72, 50, 'Teacher\nGrokking\n(~Epoch 70)',
             fontsize=10,
             color='gray',
             verticalalignment='center')

    # === Right plot: Training Loss ===
    ax2.semilogy(teacher_data['epoch'][:50],  # Only first 50 epochs for comparison
                 teacher_data['train_loss'][:50],
                 label='Teacher (128-dim)',
                 linestyle='--',
                 color='black',
                 linewidth=2.5,
                 alpha=0.7)

    ax2.semilogy(logit_data['epoch'],
                 logit_data['train_loss'],
                 label='Logit Distillation (64-dim)',
                 color='red',
                 marker='o',
                 markevery=5)

    ax2.semilogy(attention_data['epoch'],
                 attention_data['train_loss'],
                 label='Attention Transfer (64-dim)',
                 color='blue',
                 marker='s',
                 markevery=5)

    ax2.semilogy(feature_data['epoch'],
                 feature_data['train_loss'],
                 label='Feature Projection (64-dim)',
                 color='green',
                 marker='^',
                 markevery=5)

    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Training Loss (log scale)', fontsize=14)
    ax2.set_title('Training Loss: Distillation Strategies', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('paper_distillation_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_distillation_comparison.pdf', bbox_inches='tight')
    print(f"Saved: paper_distillation_comparison.png")
    print(f"Saved: paper_distillation_comparison.pdf")
    plt.close()


def plot_teacher_grokking():
    """Create detailed plot of teacher grokking dynamics."""

    teacher_data = load_history('runs/teacher_adamw')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot train and val accuracy
    ax.plot(teacher_data['epoch'],
            [acc * 100 for acc in teacher_data['train_acc']],
            label='Training Accuracy',
            color='blue',
            linewidth=2.5)

    ax.plot(teacher_data['epoch'],
            [acc * 100 for acc in teacher_data['val_acc']],
            label='Validation Accuracy',
            color='red',
            linewidth=2.5)

    # Highlight grokking transition region
    ax.axvspan(60, 90, alpha=0.2, color='yellow', label='Grokking Transition')
    ax.axvline(x=70, color='gray', linestyle='--', linewidth=2, alpha=0.6)
    ax.text(72, 50, 'Grokking\n~Epoch 70',
            fontsize=12,
            color='gray',
            fontweight='bold',
            verticalalignment='center')

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Teacher Model: Grokking Dynamics (Modular Division, p=97)',
                 fontsize=16,
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig('paper_teacher_grokking.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_teacher_grokking.pdf', bbox_inches='tight')
    print(f"Saved: paper_teacher_grokking.png")
    print(f"Saved: paper_teacher_grokking.pdf")
    plt.close()


def create_results_table():
    """Generate LaTeX table of results."""

    # Load data
    teacher_data = load_history('runs/teacher_adamw')
    logit_data = load_history('runs/distill_0.5x_logit')
    attention_data = load_history('runs/distill_0.5x_attention')
    feature_data = load_history('runs/distill_0.5x_feature')

    table = r"""
\begin{table}[h]
\centering
\caption{Distillation Results: 0.5× Student Models (64-dim, 2 layers, ~140k params)}
\label{tab:results}
\begin{tabular}{lcccc}
\hline
\textbf{Model} & \textbf{Params} & \textbf{Epochs} & \textbf{Train Acc (%%)} & \textbf{Val Acc (%%)} \\
\hline
Teacher (128-dim) & 550k & 150 & %.2f & %.2f \\
\hline
Logit Distillation & 140k & 50 & %.2f & %.2f \\
Attention Transfer & 140k & 50 & %.2f & %.2f \\
Feature Projection & 140k & 50 & %.2f & %.2f \\
\hline
\end{tabular}
\end{table}
""" % (
        teacher_data['train_acc'][-1] * 100,
        teacher_data['val_acc'][-1] * 100,
        logit_data['train_acc'][-1] * 100,
        logit_data['val_acc'][-1] * 100,
        attention_data['train_acc'][-1] * 100,
        attention_data['val_acc'][-1] * 100,
        feature_data['train_acc'][-1] * 100,
        feature_data['val_acc'][-1] * 100,
    )

    with open('paper_results_table.tex', 'w') as f:
        f.write(table)

    print("Saved: paper_results_table.tex")
    print(table)


def generate_all():
    """Generate all plots and tables for the paper."""
    print("="*60)
    print("Generating paper figures and tables...")
    print("="*60)
    print()

    try:
        plot_teacher_grokking()
        print()
        plot_comparison()
        print()
        create_results_table()
        print()
        print("="*60)
        print("✓ All figures and tables generated successfully!")
        print("="*60)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    generate_all()
