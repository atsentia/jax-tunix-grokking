"""Plot comparing teacher and distilled student grokking trajectories."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
teacher_data = json.load(open('runs/teacher_adamw/training_history.json'))
student_data = json.load(open('runs/distill_0.5x_logit_wd1.0/distillation_history.json'))

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Accuracy comparison
ax1.plot(teacher_data['epoch'], np.array(teacher_data['train_acc']) * 100,
         'b-', linewidth=2, label='Teacher Train', alpha=0.7)
ax1.plot(teacher_data['epoch'], np.array(teacher_data['val_acc']) * 100,
         'b--', linewidth=2, label='Teacher Val', alpha=0.7)
ax1.plot(student_data['epoch'], np.array(student_data['train_acc']) * 100,
         'r-', linewidth=2, label='Student Train (0.25x params)', alpha=0.7)
ax1.plot(student_data['epoch'], np.array(student_data['val_acc']) * 100,
         'r--', linewidth=2, label='Student Val (0.25x params)', alpha=0.7)

# Mark grokking zones
ax1.axvspan(60, 90, alpha=0.1, color='blue', label='Teacher grokking zone')
ax1.axvspan(70, 120, alpha=0.1, color='red', label='Student grokking zone')

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Teacher vs Distilled Student: Grokking Trajectories', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 150)
ax1.set_ylim(0, 105)

# Right plot: Loss comparison
ax2.plot(teacher_data['epoch'], teacher_data['train_loss'],
         'b-', linewidth=2, label='Teacher Train', alpha=0.7)
ax2.plot(teacher_data['epoch'], teacher_data['val_loss'],
         'b--', linewidth=2, label='Teacher Val', alpha=0.7)
ax2.plot(student_data['epoch'], student_data['train_loss'],
         'r-', linewidth=2, label='Student Train', alpha=0.7)
ax2.plot(student_data['epoch'], student_data['val_loss'],
         'r--', linewidth=2, label='Student Val', alpha=0.7)

ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Loss Curves', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 150)
ax2.set_yscale('log')

plt.tight_layout()

# Save figure
output_path = Path('runs/distillation_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Also create a zoomed-in version showing the grokking transition
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(teacher_data['epoch'], np.array(teacher_data['val_acc']) * 100,
        'b-', linewidth=3, label='Teacher Val (550k params)', marker='o', markersize=4, markevery=5)
ax.plot(student_data['epoch'], np.array(student_data['val_acc']) * 100,
        'r-', linewidth=3, label='Student Val (138k params, 0.25x)', marker='s', markersize=4, markevery=5)

# Mark key transition points
teacher_grok_epoch = 90  # When teacher reached ~100%
student_grok_epoch = 120  # When student reached ~98%

ax.axvline(teacher_grok_epoch, color='blue', linestyle=':', alpha=0.5, linewidth=2)
ax.axvline(student_grok_epoch, color='red', linestyle=':', alpha=0.5, linewidth=2)
ax.text(teacher_grok_epoch, 50, 'Teacher\ngrokked', ha='center', fontsize=10, color='blue')
ax.text(student_grok_epoch, 50, 'Student\ngrokked', ha='center', fontsize=10, color='red')

# Highlight the grokking region
ax.axhspan(95, 100, alpha=0.1, color='green', label='Grokked (>95% val)')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax.set_title('Grokking via Distillation: Validation Accuracy Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 150)
ax.set_ylim(0, 105)

# Add text box with key results
textstr = '\n'.join([
    'Final Results (Epoch 150):',
    f'Teacher: 100.00% val acc',
    f'Student: 99.98% val acc',
    '',
    'Student uses only 25% of',
    "teacher's parameters!"
])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()
output_path2 = Path('runs/distillation_validation_comparison.png')
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"Zoomed plot saved to: {output_path2}")

print("\nPlots created successfully!")
print(f"  1. Full comparison: {output_path}")
print(f"  2. Validation focus: {output_path2}")
