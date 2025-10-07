"""Create visualization plots for mechanistic analysis results."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('mechanistic_analysis_results.json', 'r') as f:
    results = json.load(f)

# Set matplotlib style
plt.style.use('seaborn-v0_8-paper')

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# === Plot 1: CKA Scores ===
strategies = ['Logit', 'Attention', 'Feature']
layers = ['Layer 0', 'Layer 1']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

x = np.arange(len(layers))
width = 0.25

for i, strategy in enumerate(['logit', 'attention', 'feature']):
    values = [results['cka_scores'][f'layer_{j}'][strategy] for j in range(2)]
    ax1.bar(x + i*width, values, width, label=strategies[i], color=colors[i])

ax1.set_ylabel('CKA Score', fontsize=12, fontweight='bold')
ax1.set_title('Representation Similarity (CKA)', fontsize=13, fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticks(x + width)
ax1.set_xticklabels(layers)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 1.0)

# === Plot 2: Effective Rank ===
x = np.arange(len(layers))
width = 0.2

models = ['Teacher', 'Logit', 'Attention', 'Feature']
model_keys = ['teacher', 'logit', 'attention', 'feature']
colors_rank = ['black', '#1f77b4', '#ff7f0e', '#2ca02c']

for i, (model, key) in enumerate(zip(models, model_keys)):
    values = [results['effective_ranks'][f'layer_{j}'][key] for j in range(2)]
    ax2.bar(x + i*width, values, width, label=model, color=colors_rank[i])

ax2.set_ylabel('Effective Rank', fontsize=12, fontweight='bold')
ax2.set_title('Hidden State Dimensionality', fontsize=13, fontweight='bold')
ax2.set_xticks(x + 1.5*width)
ax2.set_xticklabels(layers)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# === Plot 3: Attention Entropy ===
x = np.arange(len(layers))
width = 0.2

for i, (model, key) in enumerate(zip(models, model_keys)):
    values = [results['attention_entropies'][f'layer_{j}'][key] for j in range(2)]
    ax3.bar(x + i*width, values, width, label=model, color=colors_rank[i])

ax3.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold')
ax3.set_title('Attention Pattern Diversity', fontsize=13, fontweight='bold')
ax3.set_xticks(x + 1.5*width)
ax3.set_xticklabels(layers)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('paper_mechanistic_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('paper_mechanistic_analysis.pdf', bbox_inches='tight')
print("✓ Saved: paper_mechanistic_analysis.png")
print("✓ Saved: paper_mechanistic_analysis.pdf")
plt.close()

# Create summary table
print("\n--- Summary for Paper ---")
print("\nCKA Scores (Layer 0 / Layer 1):")
for strategy in ['logit', 'attention', 'feature']:
    cka_0 = results['cka_scores']['layer_0'][strategy]
    cka_1 = results['cka_scores']['layer_1'][strategy]
    print(f"  {strategy.capitalize():12s}: {cka_0:.3f} / {cka_1:.3f}")

print("\nEffective Rank (Layer 0 / Layer 1):")
for key, label in [('teacher', 'Teacher'), ('logit', 'Logit'), ('attention', 'Attention'), ('feature', 'Feature')]:
    rank_0 = results['effective_ranks']['layer_0'][key]
    rank_1 = results['effective_ranks']['layer_1'][key]
    print(f"  {label:12s}: {rank_0:.1f} / {rank_1:.1f}")
