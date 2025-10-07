"""Recursive distillation: 0.25x student from 0.5x attention-distilled teacher."""
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

# Use the attention-distilled 0.5x model as the teacher
TEACHER_CHECKPOINT = Path("runs/distill_0.5x_attention/checkpoints")
OUTPUT_DIR = Path("runs/recursive_0.25x_from_attention")
STUDENT_SCALE = 0.5  # 0.5 of 0.5x teacher = 0.25x of original

print("="*80)
print("RECURSIVE Distillation: 0.25x from Attention-Distilled 0.5x Teacher")
print("="*80)

# Load teacher (the 0.5x attention-distilled model)
print("\n1. Loading 0.5x teacher (attention-distilled)...")
teacher, teacher_metadata = load_teacher_checkpoint(TEACHER_CHECKPOINT)
teacher_config = TransformerConfig(**teacher_metadata.config)

print(f"   Teacher (0.5x distilled): {teacher_config.dim}d, ~138k params")

# Create even smaller student (0.5 * 0.5 = 0.25x of original)
print(f"\n2. Creating 0.25x student (scale={STUDENT_SCALE} of teacher)...")
student_config = scale_transformer_config(teacher_config, scale=STUDENT_SCALE, depth_scale=1.0)
print(f"   Student (0.25x overall): {student_config.dim}d, ~35k params")

# Configs - use same successful settings
data_config = DataConfig(p=97, operation="/", train_fraction=0.5, batch_size=512)
training_config = TrainingConfig(
    epochs=300,
    learning_rate=1e-3,
    weight_decay=1.0,  # Critical for grokking!
    log_every=1,
)
distill_config = DistillationConfig(
    strategies=(DistillationStrategy.ATTENTION,),  # Use same strategy as first level
    temperature=2.0,
    alpha=0.5,
    attention_weight=1.0,
)

print(f"\n3. Distillation settings:")
print(f"   - Strategy: ATTENTION (same as teacher's training)")
print(f"   - Epochs: {training_config.epochs}")
print(f"   - Learning rate: {training_config.learning_rate}")
print(f"   - Weight decay: {training_config.weight_decay}")

print(f"\n4. Starting recursive distillation...")
print(f"   Teacher chain: Original 128d → 64d (attention) → 32d (attention)")
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
print("Recursive Distillation Complete!")
print("="*80)
print(f"\nFinal metrics (epoch {result.history['epoch'][-1]}):")
print(f"  Train accuracy: {result.history['train_acc'][-1]:.2%}")
print(f"  Val accuracy: {result.final_metrics['val_acc']:.2%}")
print(f"  Train loss: {result.history['train_loss'][-1]:.4f}")
print(f"  Val loss: {result.final_metrics['val_loss']:.4f}")
print(f"\nModel size: {student_config.dim}d, ~35k params (0.25x original, 0.06x original params)")
print(f"Results saved to: {OUTPUT_DIR}")
