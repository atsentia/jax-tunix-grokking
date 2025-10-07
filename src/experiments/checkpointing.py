"""
Checkpoint management for Grokking models using Orbax.

Provides utilities for:
- Saving/loading NNX models with full architecture metadata
- Structured directory naming for experiments
- Validation of checkpoint integrity
- Handling RNG keys and dropout layers correctly
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import yaml

from models import Transformer, TransformerConfig


class CheckpointMetadata:
    """Metadata for a checkpoint, including architecture and training info."""

    def __init__(
        self,
        # Architecture
        depth: int,
        dim: int,
        heads: int,
        n_tokens: int,
        seq_len: int,
        dropout: float,
        pool: str,
        # Training context
        p: int,
        operation: str,
        train_fraction: float,
        # Training results
        epoch: int,
        step: int,
        train_acc: float,
        val_acc: float,
        train_loss: float,
        val_loss: float,
        total_params: int,
        # Optional
        grokking_epoch: Optional[int] = None,
        weight_norm: Optional[float] = None,
        timestamp: Optional[str] = None,
        notes: Optional[str] = None
    ):
        self.architecture = {
            'depth': depth,
            'dim': dim,
            'heads': heads,
            'n_tokens': n_tokens,
            'seq_len': seq_len,
            'dropout': dropout,
            'pool': pool
        }
        self.training_context = {
            'p': p,
            'operation': operation,
            'train_fraction': train_fraction
        }
        self.training_results = {
            'epoch': epoch,
            'step': step,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'total_params': total_params,
            'grokking_epoch': grokking_epoch,
            'weight_norm': weight_norm
        }
        self.timestamp = timestamp or datetime.now().isoformat()
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'architecture': self.architecture,
            'training_context': self.training_context,
            'training_results': self.training_results,
            'timestamp': self.timestamp,
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create metadata from dictionary."""
        return cls(
            # Architecture
            **data['architecture'],
            # Training context
            **data['training_context'],
            # Training results
            **data['training_results'],
            # Optional
            timestamp=data.get('timestamp'),
            notes=data.get('notes')
        )

    def save(self, path: Path):
        """Save metadata to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> 'CheckpointMetadata':
        """Load metadata from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


def create_checkpoint_dir(
    base_dir: str,
    experiment_name: str,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    create: bool = True
) -> Path:
    """
    Create a structured checkpoint directory.

    Format: {base_dir}/{experiment_name}/checkpoint_epoch_{epoch}_step_{step}/

    Args:
        base_dir: Base directory (e.g., 'runs/')
        experiment_name: Experiment identifier (e.g., 'teacher_p97_d2_dim128')
        epoch: Training epoch (optional)
        step: Training step (optional)
        create: Whether to create the directory

    Returns:
        Path to checkpoint directory
    """
    base = Path(base_dir) / experiment_name

    if epoch is not None or step is not None:
        suffix_parts = []
        if epoch is not None:
            suffix_parts.append(f"epoch_{epoch}")
        if step is not None:
            suffix_parts.append(f"step_{step}")
        checkpoint_name = f"checkpoint_{'_'.join(suffix_parts)}"
        ckpt_dir = base / checkpoint_name
    else:
        # Use timestamp if no epoch/step provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = base / f"checkpoint_{timestamp}"

    if create:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    return ckpt_dir


def save_checkpoint(
    model: Transformer,
    checkpoint_dir: Path,
    metadata: CheckpointMetadata,
    overwrite: bool = True
) -> Path:
    """
    Save a checkpoint with model state and metadata.

    Args:
        model: Transformer model to save
        checkpoint_dir: Directory to save checkpoint
        metadata: Checkpoint metadata
        overwrite: Whether to overwrite existing checkpoint

    Returns:
        Path to saved checkpoint directory
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Split model into graphdef and state
    graphdef, state = nnx.split(model)

    # Save state with Orbax
    state_dir = checkpoint_dir / 'state'
    checkpointer = ocp.StandardCheckpointer()

    save_args = ocp.args.StandardSave(state)
    checkpointer.save(state_dir, state, save_args=save_args, force=overwrite)

    # Save graphdef separately (for reconstruction)
    graphdef_path = checkpoint_dir / 'graphdef.json'
    # Note: graphdef is not easily serializable, so we save architecture params instead
    # This is handled by metadata

    # Save metadata
    metadata_path = checkpoint_dir / 'metadata.yaml'
    metadata.save(metadata_path)

    print(f"✓ Checkpoint saved to: {checkpoint_dir}")
    print(f"  - State: {state_dir}")
    print(f"  - Metadata: {metadata_path}")

    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: Path,
    rngs: Optional[nnx.Rngs] = None
) -> Tuple[Transformer, CheckpointMetadata]:
    """
    Load a checkpoint and reconstruct the model.

    Args:
        checkpoint_dir: Directory containing checkpoint
        rngs: Random number generators (if None, uses default)

    Returns:
        Tuple of (model, metadata)
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Load metadata
    metadata_path = checkpoint_dir / 'metadata.yaml'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = CheckpointMetadata.load(metadata_path)

    # Create abstract model with same architecture
    config = TransformerConfig(**metadata.architecture)

    if rngs is None:
        rngs = nnx.Rngs(params=0, dropout=0)

    # Create abstract model (without allocating memory)
    abstract_model = nnx.eval_shape(lambda: Transformer(config, rngs))
    graphdef, abstract_state = nnx.split(abstract_model)

    # Load state with Orbax
    state_dir = checkpoint_dir / 'state'
    checkpointer = ocp.StandardCheckpointer()

    restored_state = checkpointer.restore(state_dir, abstract_state)

    # Reconstruct model
    model = nnx.merge(graphdef, restored_state)

    print(f"✓ Checkpoint loaded from: {checkpoint_dir}")
    print(f"  - Architecture: depth={config.depth}, dim={config.dim}, heads={config.heads}")
    print(f"  - Training: p={metadata.training_context['p']}, "
          f"op='{metadata.training_context['operation']}'")
    print(f"  - Results: epoch={metadata.training_results['epoch']}, "
          f"val_acc={metadata.training_results['val_acc']*100:.2f}%")

    return model, metadata


def list_checkpoints(base_dir: Path, experiment_name: Optional[str] = None) -> list:
    """
    List all checkpoints in a directory.

    Args:
        base_dir: Base directory to search
        experiment_name: Optional experiment name to filter

    Returns:
        List of checkpoint directories
    """
    base_dir = Path(base_dir)

    if experiment_name:
        search_dir = base_dir / experiment_name
    else:
        search_dir = base_dir

    if not search_dir.exists():
        return []

    # Find all directories containing metadata.yaml
    checkpoints = []
    for path in search_dir.rglob('metadata.yaml'):
        checkpoints.append(path.parent)

    return sorted(checkpoints)


def get_latest_checkpoint(base_dir: Path, experiment_name: str) -> Optional[Path]:
    """
    Get the most recent checkpoint for an experiment.

    Args:
        base_dir: Base directory
        experiment_name: Experiment name

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoints = list_checkpoints(base_dir, experiment_name)

    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return checkpoints[0]


def validate_checkpoint(checkpoint_dir: Path) -> bool:
    """
    Validate that a checkpoint is complete and loadable.

    Args:
        checkpoint_dir: Directory containing checkpoint

    Returns:
        True if checkpoint is valid, False otherwise
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Check directory exists
    if not checkpoint_dir.exists():
        print(f"✗ Checkpoint directory not found: {checkpoint_dir}")
        return False

    # Check metadata exists
    metadata_path = checkpoint_dir / 'metadata.yaml'
    if not metadata_path.exists():
        print(f"✗ Metadata file not found: {metadata_path}")
        return False

    # Check state directory exists
    state_dir = checkpoint_dir / 'state'
    if not state_dir.exists():
        print(f"✗ State directory not found: {state_dir}")
        return False

    # Try loading metadata
    try:
        metadata = CheckpointMetadata.load(metadata_path)
    except Exception as e:
        print(f"✗ Failed to load metadata: {e}")
        return False

    # Try loading checkpoint
    try:
        model, _ = load_checkpoint(checkpoint_dir)
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False

    print(f"✓ Checkpoint is valid: {checkpoint_dir}")
    return True


if __name__ == "__main__":
    # Example usage
    print("Testing checkpoint utilities...")

    # Create a test model
    config = TransformerConfig(
        depth=2,
        dim=128,
        heads=1,
        n_tokens=99,
        seq_len=4,
        dropout=0.2,
        pool='cls'
    )

    rngs = nnx.Rngs(params=42, dropout=0)
    model = Transformer(config, rngs)

    # Create metadata
    metadata = CheckpointMetadata(
        # Architecture
        depth=2, dim=128, heads=1, n_tokens=99, seq_len=4, dropout=0.2, pool='cls',
        # Training context
        p=97, operation='/', train_fraction=0.5,
        # Training results
        epoch=100, step=10000, train_acc=0.998, val_acc=0.995,
        train_loss=0.05, val_loss=0.08, total_params=554000,
        grokking_epoch=75, weight_norm=125.3,
        notes="Test checkpoint"
    )

    # Save checkpoint
    ckpt_dir = create_checkpoint_dir(
        base_dir='runs/test',
        experiment_name='test_experiment',
        epoch=100
    )

    save_checkpoint(model, ckpt_dir, metadata)

    # Load checkpoint
    loaded_model, loaded_metadata = load_checkpoint(ckpt_dir)

    # Validate
    is_valid = validate_checkpoint(ckpt_dir)

    print(f"\n✓ Checkpoint test completed! Valid: {is_valid}")
