"""Simple distillation test script with detailed logging."""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
from distillation import (
    load_teacher_checkpoint,
    run_distillation,
    DataConfig,
    TrainingConfig,
    DistillationConfig,
    DistillationStrategy,
    scale_transformer_config,
)
from models import TransformerConfig
import json

# Configuration
TEACHER_CHECKPOINT = Path("runs/teacher_adamw/checkpoints")
OUTPUT_DIR = Path("runs/distill_0.5x_logit_wd1.0")
STUDENT_SCALE = 0.5

print("="*80)
print("Distillation Experiment: 0.5x Student with Logit Strategy")
print("="*80)

# Load teacher
print("\n1. Loading teacher...")
teacher, teacher_metadata = load_teacher_checkpoint(TEACHER_CHECKPOINT)
teacher_config = TransformerConfig(**teacher_metadata.config)

print(f"   Teacher config:")
print(f"     - Depth: {teacher_config.depth}")
print(f"     - Dim: {teacher_config.dim}")
print(f"     - Heads: {teacher_config.heads}")
print(f"     - Parameters: {550_280:,}")  # We know this from training

# Create student config (scaled)
print(f"\n2. Creating student (scale={STUDENT_SCALE})...")
student_config = scale_transformer_config(teacher_config, scale=STUDENT_SCALE, depth_scale=1.0)

print(f"   Student config:")
print(f"     - Depth: {student_config.depth}")
print(f"     - Dim: {student_config.dim}")
print(f"     - Heads: {student_config.heads}")

# Calculate student parameters
# Rough estimate: params scale with dim^2 (attention) + dim (norms/embeddings)
# Teacher: 550k with dim=128
# Student: dim=64 (half), so roughly 550k * (64/128)^2 ≈ 137k
student_param_ratio = (student_config.dim / teacher_config.dim) ** 2
estimated_student_params = int(550_280 * student_param_ratio)
print(f"     - Estimated parameters: ~{estimated_student_params:,}")

# Configs
data_config = DataConfig(p=97, operation="/", train_fraction=0.5, batch_size=512)
training_config = TrainingConfig(
    epochs=150,  # Match teacher training length
    learning_rate=1e-3,  # Match teacher LR
    weight_decay=1.0,  # CRITICAL: Match teacher weight decay for grokking!
    log_every=1,
)
distill_config = DistillationConfig(
    strategies=(DistillationStrategy.LOGIT,),
    temperature=2.0,
    alpha=0.5,  # 50% hard labels, 50% soft labels from teacher
)

print(f"\n3. Distillation settings:")
print(f"   - Strategy: Logit-only (soft targets from teacher)")
print(f"   - Temperature: {distill_config.temperature}")
print(f"   - Alpha: {distill_config.alpha} (balance between hard/soft targets)")
print(f"   - Epochs: {training_config.epochs}")
print(f"   - Learning rate: {training_config.learning_rate}")
print(f"   - Weight decay: {training_config.weight_decay} (CRITICAL for grokking!)")

print(f"\n4. Starting distillation...")
print("   (Progress will be logged every epoch)")
print("="*80)

# Run distillation
result = run_distillation(
    teacher_checkpoint=TEACHER_CHECKPOINT,
    student_config=student_config,
    data_config=data_config,
    training_config=training_config,
    distill_config=distill_config,
    output_dir=OUTPUT_DIR,
    seed=42,
)

print("\n" + "="*80)
print("Distillation Complete!")
print("="*80)
print(f"\nFinal metrics (epoch {result.history['epoch'][-1]}):")
print(f"  Train accuracy: {result.history['train_acc'][-1]:.2%}")
print(f"  Val accuracy: {result.final_metrics['val_acc']:.2%}")
print(f"  Train loss: {result.history['train_loss'][-1]:.4f}")
print(f"  Val loss: {result.final_metrics['val_loss']:.4f}")

print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"  - distillation_history.json")
print(f"  - checkpoints/")
print(f"\nCompare to teacher:")
print(f"  Teacher had 100% val accuracy after grokking")
