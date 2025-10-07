"""
Training script for Grokking with NNX models (baseline-equivalent training loop).

This script uses a manual training loop (NOT Tunix trainer) to match the baseline
exactly and verify optimizer parity. Once validated, we can integrate with Tunix.

Critical parameters for grokking (per PRD):
- AdamW: lr=1e-3, β1=0.9, β2=0.98, weight_decay=1.0
- Linear warmup: 10 steps
- Architecture: depth=2, dim=128, heads=1, dropout=0.2
"""

import dataclasses
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from typing import Dict, Tuple
import argparse
from pathlib import Path

from checkpointing import (
    CheckpointMetadata,
    load_history,
    restore_checkpoint,
    save_checkpoint,
    save_history,
    read_metadata,
)
from models import Transformer, TransformerConfig
from data import grokking_data
from optimizers import create_optimizer as create_optimizer_factory


def create_optimizer(
    optimizer_type: str,
    learning_rate: float,
    warmup_steps: int,
    beta1: float,
    beta2: float,
    weight_decay: float
) -> optax.GradientTransformation:
    """
    Create optimizer with linear warmup.

    Supports both AdamW (original) and Muon (newer) optimizers.

    Args:
        optimizer_type: 'adamw' or 'muon'
        learning_rate: Peak learning rate
        warmup_steps: Linear warmup steps
        beta1: AdamW beta1 (ignored for Muon)
        beta2: AdamW beta2 (ignored for Muon)
        weight_decay: Weight decay coefficient

    Returns:
        Optax GradientTransformation
    """
    return create_optimizer_factory(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2
    )


def compute_loss_and_accuracy(
    model: Transformer,
    batch_X: jax.Array,
    batch_y: jax.Array,
    n_tokens: int,
    training: bool = True
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute loss and accuracy for a batch."""
    # Forward pass
    logits = model(batch_X, training=training)

    # Cross-entropy loss
    one_hot = jax.nn.one_hot(batch_y, n_tokens)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()

    # Accuracy
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean((preds == batch_y).astype(jnp.float32))

    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }

    return loss, metrics


def train_step(
    model: Transformer,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch_X: jax.Array,
    batch_y: jax.Array,
    n_tokens: int,
    dropout_rng: jax.Array
) -> Tuple[Transformer, optax.OptState, Dict[str, jax.Array]]:
    """Single training step (simple, working pattern from PoC)."""

    # Define loss function
    loss_fn = lambda m: compute_loss_and_accuracy(m, batch_X, batch_y, n_tokens, training=True)

    # Compute gradients (simple NNX pattern)
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Extract gradient and parameter states (only trainable Params)
    grad_state = nnx.state(grads, nnx.Param)
    param_state = nnx.state(model, nnx.Param)

    # Apply optimizer updates
    updates, opt_state = optimizer.update(grad_state, opt_state, param_state)
    new_param_state = optax.apply_updates(param_state, updates)

    # Update model with new parameters
    nnx.update(model, new_param_state)

    return model, opt_state, metrics


def eval_step(
    model_state: nnx.State,
    graphdef: nnx.GraphDef,
    batch_X: jax.Array,
    batch_y: jax.Array,
    n_tokens: int
) -> Dict[str, jax.Array]:
    """Evaluation step (non-JIT for simplicity, can JIT later with static_argnums)."""
    model = nnx.merge(graphdef, model_state)
    _, metrics = compute_loss_and_accuracy(model, batch_X, batch_y, n_tokens, training=False)
    return metrics


def compute_weight_norm(model: Transformer) -> float:
    """Compute L2 norm of all model parameters."""
    # Use graphdef/state split to get clean parameters
    graphdef, state = nnx.split(model)

    def is_param_array(x):
        """Check if x is a valid parameter array (not RNG key)."""
        if not isinstance(x, jax.Array):
            return False
        # Exclude RNG keys (they have special dtypes)
        try:
            _ = x + 0  # This will fail for RNG keys
            return True
        except:
            return False

    param_arrays = [p for p in jax.tree.leaves(state) if is_param_array(p)]
    total_norm = jnp.sqrt(sum(jnp.sum(p.astype(jnp.float32)**2) for p in param_arrays))
    return float(total_norm)


def create_history_dict() -> Dict[str, list]:
    """Create an empty history dictionary for metrics tracking."""

    return {
        'step': [],
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'weight_norm': [],
    }


def train(
    # Model config
    depth: int = 2,
    dim: int = 128,
    heads: int = 1,
    dropout: float = 0.2,
    pool: str = 'cls',

    # Data config
    p: int = 97,
    operation: str = '/',
    train_fraction: float = 0.5,
    batch_size: int = 512,

    # Training config
    epochs: int = 150,
    learning_rate: float = 1e-3,
    weight_decay: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.98,
    warmup_steps: int = 10,
    optimizer_type: str = 'adamw',

    # Other
    seed: int = 42,
    log_every: int = 10,
    save_dir: str = None,
    checkpoint_dir: str = None,
    resume: bool = False,
    max_steps: int = None
):
    """
    Train grokking model with NNX.

    Matches baseline hyper-parameters while adding Orbax-based checkpointing
    and resume support for long-running experiments.
    """

    print("=" * 80)
    print("Grokking Training - NNX Implementation")
    print("=" * 80)

    # Configuration summary
    print(f"\nConfiguration:")
    print(f"  Model: depth={depth}, dim={dim}, heads={heads}, dropout={dropout}")
    print(f"  Data: p={p}, op='{operation}', train_frac={train_fraction}")
    print(f"  Training: epochs={epochs}, batch_size={batch_size}")
    print(f"  Optimizer: {optimizer_type}(lr={learning_rate}, wd={weight_decay}, β1={beta1}, β2={beta2})")
    print(f"  Warmup: {warmup_steps} steps")
    print(f"  Seed: {seed}")
    if save_dir:
        print(f"  Save dir: {save_dir}")
    if checkpoint_dir:
        print(f"  Checkpoints: {checkpoint_dir} (resume={'yes' if resume else 'no'})")

    # 1. Prepare data
    print(f"\n1. Loading data...")
    X_train, y_train, X_val, y_val = grokking_data(
        p=p, op=operation, train_fraction=train_fraction, seed=seed
    )
    seq_len = X_train.shape[1]
    n_tokens = p + 2

    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val: {X_val.shape[0]} samples")
    print(f"   Vocab size: {n_tokens}, Seq len: {seq_len}")

    # 2. Create model
    print(f"\n2. Creating model...")
    config = TransformerConfig(
        depth=depth,
        dim=dim,
        heads=heads,
        n_tokens=n_tokens,
        seq_len=seq_len,
        dropout=dropout,
        pool=pool
    )

    rngs = nnx.Rngs(params=seed, dropout=seed)
    model = Transformer(config, rngs)

    save_path = Path(save_dir) if save_dir else None
    history_file = None
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        history_file = save_path / 'training_history.json'

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
    metadata = None
    optimizer_config = {
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
    }
    config_dict = dataclasses.asdict(config)
    if checkpoint_path:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        metadata = read_metadata(checkpoint_path)
        if metadata is None:
            metadata = CheckpointMetadata(
                config=config_dict,
                optimizer=optimizer_config,
                seed=seed,
            )
        else:
            if metadata.config and metadata.config != config_dict:
                raise ValueError(
                    "Checkpoint configuration does not match current run settings"
                )
            metadata.config = config_dict
            metadata.optimizer = optimizer_config
            metadata.seed = seed

    param_count = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
    print(f"   Parameters: {param_count:,}")
    print(f"   Initial weight norm: {compute_weight_norm(model):.4f}")

    # 3. Create optimizer
    print(f"\n3. Creating optimizer...")
    print(f"   Optimizer type: {optimizer_type}")
    optimizer = create_optimizer(optimizer_type, learning_rate, warmup_steps, beta1, beta2, weight_decay)

    # Initialize optimizer state with only trainable parameters (nnx.Param)
    param_state = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(param_state)

    # Store schedule for LR tracking
    warmup_fn = optax.linear_schedule(0.0, learning_rate, warmup_steps)
    constant_fn = optax.constant_schedule(learning_rate)
    schedule_fn = optax.join_schedules([warmup_fn, constant_fn], [warmup_steps])

    # 4. Training loop
    print(f"\n4. Starting training...")
    print(f"{'='*80}")

    num_train = X_train.shape[0]
    num_batches = int(np.ceil(num_train / batch_size))
    total_steps = 0
    start_epoch = 1

    history = create_history_dict()
    if resume and history_file and history_file.exists():
        history = load_history(history_file)
        if not history:
            history = create_history_dict()

    rng = jax.random.PRNGKey(seed)

    if checkpoint_path and metadata and resume:
        restored = restore_checkpoint(
            checkpoint_path,
            model,
            opt_state,
            rng,
            metadata,
        )
        if restored:
            opt_state = restored.optimizer_state
            rng = restored.rng_key
            total_steps = restored.step
            start_epoch = restored.epoch + 1
            if history_file and history_file.exists():
                history = load_history(history_file)
                if not history:
                    history = create_history_dict()
            print(
                f"  Resumed from step {total_steps} (epoch {restored.epoch})"
            )
        else:
            print("  No checkpoint found; starting fresh.")

    for epoch in range(start_epoch, epochs + 1):
        if max_steps and total_steps >= max_steps:
            break

        # Shuffle training data
        perm = np.random.permutation(num_train)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_start = time.time()

        for batch_idx in range(num_batches):
            if max_steps and total_steps >= max_steps:
                break

            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_train)
            batch_X = X_train_shuffled[start_idx:end_idx]
            batch_y = y_train_shuffled[start_idx:end_idx]

            # Training step
            rng, dropout_rng = jax.random.split(rng)
            model, opt_state, train_metrics = train_step(
                model, optimizer, opt_state, batch_X, batch_y, n_tokens, dropout_rng
            )

            epoch_loss += float(train_metrics['loss']) * batch_X.shape[0]
            epoch_acc += float(train_metrics['accuracy']) * batch_X.shape[0]
            total_steps += 1

        # Epoch metrics
        train_loss = epoch_loss / num_train
        train_acc = epoch_acc / num_train

        # Validation
        graphdef, model_state = nnx.split(model)
        val_metrics = eval_step(model_state, graphdef, X_val, y_val, n_tokens)
        val_loss = float(val_metrics['loss'])
        val_acc = float(val_metrics['accuracy'])

        # Get current LR and weight norm
        current_lr = schedule_fn(total_steps) if 'schedule_fn' in locals() else learning_rate
        weight_norm = compute_weight_norm(model)

        # Log
        if epoch % log_every == 0 or epoch == 1:
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:3d}/{epochs} ({total_steps:5d} steps, {epoch_time:.1f}s): "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc*100:.2f}%, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%, "
                  f"lr={current_lr:.6f}, w_norm={weight_norm:.2f}")

        # Save history
        history['step'].append(total_steps)
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(float(current_lr))
        history['weight_norm'].append(weight_norm)

        if history_file:
            save_history(history_file, history)

        if checkpoint_path and metadata:
            save_checkpoint(
                checkpoint_path,
                model,
                opt_state,
                rng,
                total_steps,
                epoch,
                metadata,
            )

        # Check for assertions
        assert np.isfinite(train_loss), f"Train loss not finite at epoch {epoch}"
        assert np.isfinite(val_loss), f"Val loss not finite at epoch {epoch}"

    print(f"{'='*80}")
    print(f"Training complete!")
    print(f"  Total steps: {total_steps}")
    print(f"  Final train acc: {train_acc*100:.2f}%")
    print(f"  Final val acc: {val_acc*100:.2f}%")
    print(f"  Final weight norm: {weight_norm:.2f}")

    # Save results
    if history_file:
        save_history(history_file, history)
        print(f"\n  Saved history to: {history_file}")

    if checkpoint_path and metadata and metadata.latest_step is not None:
        latest_dir = checkpoint_path / f"step_{metadata.latest_step:08d}"
        print(f"  Latest checkpoint: {latest_dir}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train grokking model with NNX')

    # Model args
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Data args
    parser.add_argument('--p', type=int, default=97)
    parser.add_argument('--operation', type=str, default='/')
    parser.add_argument('--train_fraction', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=512)

    # Training args
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'muon'],
                        help='Optimizer type: adamw (original) or muon (newer)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='runs/nnx_baseline')
    parser.add_argument('--checkpoint_dir', type=str, default='runs/nnx_baseline/checkpoints')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()

    # Map CLI args to function params
    train_args = vars(args)
    train_args['optimizer_type'] = train_args.pop('optimizer')

    model, history = train(**train_args)
