"""Test attention transfer distillation strategy."""
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

TEACHER_CHECKPOINT = Path("runs/teacher_adamw/checkpoints")
OUTPUT_DIR = Path("runs/distill_0.5x_attention")
STUDENT_SCALE = 0.5

print("="*80)
print("Distillation Experiment: ATTENTION TRANSFER Strategy")
print("="*80)

# Load teacher
print("\n1. Loading teacher...")
teacher, teacher_metadata = load_teacher_checkpoint(TEACHER_CHECKPOINT)
teacher_config = TransformerConfig(**teacher_metadata.config)

print(f"   Teacher: {teacher_config.dim} dims, ~550k params")

# Create student
print(f"\n2. Creating student (scale={STUDENT_SCALE})...")
student_config = scale_transformer_config(teacher_config, scale=STUDENT_SCALE, depth_scale=1.0)
print(f"   Student: {student_config.dim} dims, ~138k params")

# Configs
data_config = DataConfig(p=97, operation="/", train_fraction=0.5, batch_size=512)
training_config = TrainingConfig(
    epochs=150,
    learning_rate=1e-3,
    weight_decay=1.0,  # Critical for grokking!
    log_every=1,
)
distill_config = DistillationConfig(
    strategies=(DistillationStrategy.ATTENTION,),  # Attention transfer only
    temperature=2.0,
    alpha=0.5,
    attention_weight=1.0,
)

print(f"\n3. Distillation settings:")
print(f"   - Strategy: ATTENTION TRANSFER (match attention patterns)")
print(f"   - Attention weight: {distill_config.attention_weight}")
print(f"   - Epochs: {training_config.epochs}")
print(f"   - Learning rate: {training_config.learning_rate}")
print(f"   - Weight decay: {training_config.weight_decay}")

print(f"\n4. Starting distillation...")
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
