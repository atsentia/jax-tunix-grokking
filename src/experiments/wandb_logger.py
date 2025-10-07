"""
Weights & Biases logging utilities for grokking experiments.

Provides:
- Experiment initialization with proper grouping
- Metric logging for training/validation
- Artifact logging for checkpoints and plots
- Comparison utilities for distillation experiments
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


class WandbLogger:
    """Wrapper for W&B logging with grokking-specific utilities."""

    def __init__(
        self,
        project: str = "grokking-distillation",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = "online",
        enabled: bool = True
    ):
        """
        Initialize W&B logger.

        Args:
            project: W&B project name
            entity: W&B entity (username or team)
            name: Run name
            group: Group name for organizing related runs
            tags: List of tags for the run
            config: Configuration dictionary to log
            mode: W&B mode ('online', 'offline', 'disabled')
            enabled: Whether to enable logging
        """
        self.enabled = enabled and WANDB_AVAILABLE

        if not WANDB_AVAILABLE:
            print("W&B logging disabled: wandb not installed")
            self.enabled = False

        if self.enabled:
            # Check if already initialized
            if wandb.run is not None:
                print("W&B already initialized, using existing run")
                self.run = wandb.run
            else:
                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    group=group,
                    tags=tags,
                    config=config,
                    mode=mode
                )
        else:
            self.run = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number (epoch or training step)
            commit: Whether to commit immediately
        """
        if not self.enabled:
            return

        wandb.log(metrics, step=step, commit=commit)

    def log_training_metrics(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        weight_norm: Optional[float] = None,
        commit: bool = True
    ):
        """
        Log standard training metrics.

        Args:
            epoch: Current epoch
            step: Current training step
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            lr: Learning rate
            weight_norm: Model weight norm (optional)
            commit: Whether to commit immediately
        """
        metrics = {
            'epoch': epoch,
            'step': step,
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'lr': lr
        }

        if weight_norm is not None:
            metrics['weight_norm'] = weight_norm

        self.log(metrics, step=step, commit=commit)

    def log_distillation_metrics(
        self,
        epoch: int,
        step: int,
        student_train_loss: float,
        student_train_acc: float,
        student_val_loss: float,
        student_val_acc: float,
        teacher_val_acc: float,
        distill_loss: float,
        task_loss: Optional[float] = None,
        lr: float = None,
        commit: bool = True
    ):
        """
        Log distillation-specific metrics.

        Args:
            epoch: Current epoch
            step: Current training step
            student_train_loss: Student training loss
            student_train_acc: Student training accuracy
            student_val_loss: Student validation loss
            student_val_acc: Student validation accuracy
            teacher_val_acc: Teacher validation accuracy (for comparison)
            distill_loss: Distillation loss component
            task_loss: Task loss component (optional)
            lr: Learning rate (optional)
            commit: Whether to commit immediately
        """
        metrics = {
            'epoch': epoch,
            'step': step,
            'student/train_loss': student_train_loss,
            'student/train_acc': student_train_acc,
            'student/val_loss': student_val_loss,
            'student/val_acc': student_val_acc,
            'teacher/val_acc': teacher_val_acc,
            'distillation/loss': distill_loss,
            'comparison/acc_gap': teacher_val_acc - student_val_acc
        }

        if task_loss is not None:
            metrics['distillation/task_loss'] = task_loss

        if lr is not None:
            metrics['lr'] = lr

        self.log(metrics, step=step, commit=commit)

    def log_grokking_event(self, epoch: int, step: int, val_acc: float):
        """
        Log a grokking event (sudden jump in validation accuracy).

        Args:
            epoch: Epoch when grokking occurred
            step: Step when grokking occurred
            val_acc: Validation accuracy at grokking
        """
        if not self.enabled:
            return

        self.log({
            'grokking/epoch': epoch,
            'grokking/step': step,
            'grokking/val_acc': val_acc
        }, commit=True)

        # Also add a summary metric
        wandb.run.summary['grokking_epoch'] = epoch
        wandb.run.summary['grokking_val_acc'] = val_acc

    def log_checkpoint(self, checkpoint_path: Path, metadata: Dict[str, Any], aliases: Optional[List[str]] = None):
        """
        Log a checkpoint as a W&B artifact.

        Args:
            checkpoint_path: Path to checkpoint directory
            metadata: Checkpoint metadata
            aliases: Aliases for the artifact (e.g., ['latest', 'best'])
        """
        if not self.enabled:
            return

        checkpoint_path = Path(checkpoint_path)

        artifact = wandb.Artifact(
            name=f"checkpoint-{checkpoint_path.parent.name}",
            type="model",
            metadata=metadata
        )

        artifact.add_dir(str(checkpoint_path))

        wandb.log_artifact(artifact, aliases=aliases or ['latest'])

    def log_plot(self, fig: plt.Figure, name: str, commit: bool = True):
        """
        Log a matplotlib figure to W&B.

        Args:
            fig: Matplotlib figure
            name: Name for the plot
            commit: Whether to commit immediately
        """
        if not self.enabled:
            return

        self.log({name: wandb.Image(fig)}, commit=commit)

    def log_training_curve(self, history: Dict[str, List], save_path: Optional[Path] = None):
        """
        Log training curves (train/val loss and accuracy).

        Args:
            history: Training history dictionary
            save_path: Optional path to save plot
        """
        if not self.enabled:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss')
        axes[0].plot(history['epoch'], history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[1].plot(history['epoch'], [a*100 for a in history['train_acc']], label='Train Acc')
        axes[1].plot(history['epoch'], [a*100 for a in history['val_acc']], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        self.log_plot(fig, "training_curves")
        plt.close(fig)

    def log_distillation_comparison(
        self,
        history: Dict[str, List],
        teacher_val_acc: float,
        save_path: Optional[Path] = None
    ):
        """
        Log distillation comparison curves (student vs teacher).

        Args:
            history: Student training history
            teacher_val_acc: Teacher validation accuracy (constant)
            save_path: Optional path to save plot
        """
        if not self.enabled:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy comparison
        epochs = history['epoch']
        axes[0].plot(epochs, [a*100 for a in history['val_acc']], label='Student Val Acc', linewidth=2)
        axes[0].axhline(y=teacher_val_acc*100, color='r', linestyle='--', label='Teacher Val Acc', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Student vs Teacher Validation Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy gap
        acc_gap = [(teacher_val_acc - student_acc)*100 for student_acc in history['val_acc']]
        axes[1].plot(epochs, acc_gap, color='purple', linewidth=2)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy Gap (%)')
        axes[1].set_title('Teacher-Student Accuracy Gap')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        self.log_plot(fig, "distillation_comparison")
        plt.close(fig)

    def log_parameter_efficiency(
        self,
        model_name: str,
        params: int,
        val_acc: float,
        commit: bool = True
    ):
        """
        Log parameter efficiency metric (accuracy per million parameters).

        Args:
            model_name: Model identifier
            params: Parameter count
            val_acc: Validation accuracy
            commit: Whether to commit immediately
        """
        if not self.enabled:
            return

        acc_per_million = val_acc / (params / 1e6)

        self.log({
            f'{model_name}/params': params,
            f'{model_name}/val_acc': val_acc,
            f'{model_name}/acc_per_m_params': acc_per_million
        }, commit=commit)

    def finish(self):
        """Finish the W&B run."""
        if self.enabled and self.run is not None:
            wandb.finish()


def create_experiment_name(
    experiment_type: str,
    p: int,
    operation: str,
    architecture: str,
    suffix: Optional[str] = None
) -> str:
    """
    Create a standardized experiment name.

    Args:
        experiment_type: Type of experiment (e.g., 'grokking', 'distill_shrink')
        p: Prime modulus
        operation: Arithmetic operation
        architecture: Architecture description (e.g., 'd2_dim128')
        suffix: Optional suffix

    Returns:
        Experiment name string
    """
    name = f"{experiment_type}_p{p}_{operation}_{architecture}"

    if suffix:
        name = f"{name}_{suffix}"

    return name


def create_experiment_group(
    experiment_type: str,
    teacher_name: Optional[str] = None
) -> str:
    """
    Create a standardized experiment group name.

    Args:
        experiment_type: Type of experiment
        teacher_name: Teacher model name (for distillation experiments)

    Returns:
        Group name string
    """
    if teacher_name:
        return f"{experiment_type}_{teacher_name}"
    else:
        return experiment_type


if __name__ == "__main__":
    # Test W&B logger
    print("Testing W&B logger...")

    # Create logger in disabled mode for testing
    logger = WandbLogger(
        project="test-grokking",
        name="test-run",
        group="test-group",
        tags=["test"],
        mode="disabled",
        enabled=False
    )

    # Test metric logging
    logger.log_training_metrics(
        epoch=10,
        step=100,
        train_loss=0.5,
        train_acc=0.8,
        val_loss=0.6,
        val_acc=0.75,
        lr=0.001,
        weight_norm=120.5
    )

    # Test grokking event
    logger.log_grokking_event(epoch=75, step=7500, val_acc=0.95)

    # Test experiment naming
    exp_name = create_experiment_name(
        experiment_type="grokking",
        p=97,
        operation="div",
        architecture="d2_dim128"
    )
    print(f"Experiment name: {exp_name}")

    exp_group = create_experiment_group("distill_shrink", "teacher_p97")
    print(f"Experiment group: {exp_group}")

    logger.finish()

    print("✓ W&B logger test completed!")
