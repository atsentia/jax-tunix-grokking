"""Mechanistic interpretability analysis of distilled grokking models.

Quick wins: CKA, weight statistics, attention entropy, hidden state rank.
"""

import json
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from src.checkpointing import restore_checkpoint, read_metadata
from src.data import grokking_data
from src.models import Transformer, TransformerConfig


def centered_kernel_alignment(X: jax.Array, Y: jax.Array) -> float:
    """Compute CKA between two representation matrices.

    Args:
        X: First representation matrix [n_samples, d1]
        Y: Second representation matrix [n_samples, d2]

    Returns:
        CKA score (0 to 1, higher = more similar)
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    XY = jnp.linalg.norm(X_centered.T @ Y_centered, 'fro') ** 2
    XX = jnp.linalg.norm(X_centered.T @ X_centered, 'fro')
    YY = jnp.linalg.norm(Y_centered.T @ Y_centered, 'fro')

    return float(XY / (XX * YY))


def compute_weight_statistics(model: Transformer) -> Dict[str, float]:
    """Compute L0, L1, L2 norms and sparsity of model weights."""
    params = nnx.state(model, nnx.Param)
    weights = jax.tree_util.tree_leaves(params)

    # Filter out non-array leaves (if any)
    weights = [w for w in weights if isinstance(w, jax.Array)]

    l0 = sum(int(jnp.sum(jnp.abs(w) > 1e-6)) for w in weights)
    l1 = sum(float(jnp.sum(jnp.abs(w))) for w in weights)
    l2 = float(jnp.sqrt(sum(jnp.sum(w**2) for w in weights)))

    total_params = sum(w.size for w in weights)
    sparsity = 1.0 - (l0 / total_params)

    return {
        "L0": l0,
        "L1": l1,
        "L2": l2,
        "total_params": total_params,
        "sparsity": sparsity,
    }


def attention_entropy(attn_map: jax.Array) -> float:
    """Compute Shannon entropy of attention distribution.

    Args:
        attn_map: Attention weights [batch, heads, seq, seq]

    Returns:
        Mean entropy across all positions
    """
    eps = 1e-10
    entropy = -jnp.sum(attn_map * jnp.log(attn_map + eps), axis=-1)
    return float(entropy.mean())


def effective_rank(matrix: jax.Array, threshold: float = 0.01) -> float:
    """Compute effective rank of a matrix using singular values.

    Effective rank = (sum of singular values)^2 / (sum of squared singular values)

    Args:
        matrix: Input matrix [n, d]
        threshold: Singular values below this are considered zero

    Returns:
        Effective rank (between 1 and min(n, d))
    """
    # Flatten batch dimension if present
    if matrix.ndim > 2:
        matrix = matrix.reshape(-1, matrix.shape[-1])

    U, s, Vt = jnp.linalg.svd(matrix, full_matrices=False)

    # Filter small singular values
    s_filtered = s[s > threshold * s[0]]

    # Compute effective rank
    eff_rank = (jnp.sum(s_filtered) ** 2) / jnp.sum(s_filtered ** 2)

    return float(eff_rank)


def load_model_from_checkpoint(checkpoint_dir: Path) -> Transformer:
    """Load a transformer model from an Orbax checkpoint.

    Handles both plain Transformer checkpoints and DistillationContainer checkpoints.
    """
    import optax
    from src.optimizers import create_optimizer
    from src.distillation import DistillationContainer

    metadata = read_metadata(checkpoint_dir)
    if metadata is None:
        raise ValueError(f"No metadata found in {checkpoint_dir}")

    config = TransformerConfig(**metadata.config)
    rngs = nnx.Rngs(params=metadata.seed, dropout=metadata.seed)
    model = Transformer(config, rngs)

    # Create dummy optimizer state for restoration
    optimizer = create_optimizer(
        optimizer_type=metadata.optimizer.get("type", "adamw"),
        learning_rate=metadata.optimizer.get("learning_rate", 1e-3),
        warmup_steps=metadata.optimizer.get("warmup_steps", 0),
        beta1=metadata.optimizer.get("beta1", 0.9),
        beta2=metadata.optimizer.get("beta2", 0.98),
        weight_decay=metadata.optimizer.get("weight_decay", 0.0),
    )

    # Try loading as plain Transformer first
    opt_state = optimizer.init(nnx.state(model, nnx.Param))
    rng = jax.random.PRNGKey(metadata.seed)

    try:
        restored = restore_checkpoint(checkpoint_dir, model, opt_state, rng, metadata)
        if restored is not None:
            return model
    except (ValueError, KeyError) as e:
        # If that fails, try loading as DistillationContainer
        if "student" in str(e) or "projector" in str(e):
            print(f"  Note: Loading distillation checkpoint, extracting student...")

            # Need to create projector if feature distillation was used
            from src.distillation import FeatureProjector

            if "projector" in str(e):
                # Feature distillation - need to create projector
                # Load a dummy sample to determine number of hidden states
                from src.data import grokking_data
                X_sample, _, _, _ = grokking_data(p=97, op="/", train_fraction=0.5, seed=0)

                # Temporarily create teacher to get number of features
                teacher_config = TransformerConfig(depth=2, dim=128, heads=1, n_tokens=99, seq_len=5)
                teacher_rngs = nnx.Rngs(params=0, dropout=0)
                temp_teacher = Transformer(teacher_config, teacher_rngs)
                _, teacher_aux = temp_teacher(X_sample[:1], training=False, return_intermediates=True)
                num_features = len(teacher_aux.get("hidden_states", []))
                teacher_dim = 128

                projector = FeatureProjector(teacher_dim, config.dim, num_features, rngs)
            else:
                projector = None

            container = DistillationContainer(model, projector=projector)
            opt_state = optimizer.init(nnx.state(container, nnx.Param))
            restored = restore_checkpoint(checkpoint_dir, container, opt_state, rng, metadata)
            if restored is None:
                raise ValueError(f"Failed to restore checkpoint from {checkpoint_dir}")
            return container.student
        else:
            raise

    raise ValueError(f"Failed to restore checkpoint from {checkpoint_dir}")


def analyze_models():
    """Run all quick-win mechanistic interpretability analyses."""

    print("="*70)
    print("MECHANISTIC INTERPRETABILITY ANALYSIS")
    print("="*70)
    print()

    # Load models
    print("[1/5] Loading models...")
    teacher_dir = Path("runs/teacher_adamw/checkpoints")
    logit_dir = Path("runs/distill_0.5x_logit_wd1.0/checkpoints")
    attention_dir = Path("runs/distill_0.5x_attention/checkpoints")
    feature_dir = Path("runs/distill_0.5x_feature/checkpoints")

    teacher = load_model_from_checkpoint(teacher_dir)
    student_logit = load_model_from_checkpoint(logit_dir)
    student_attention = load_model_from_checkpoint(attention_dir)
    student_feature = load_model_from_checkpoint(feature_dir)

    print("✓ Models loaded successfully")
    print()

    # Load validation data
    print("[2/5] Loading validation data...")
    X_val, y_val, _, _ = grokking_data(p=97, op="/", train_fraction=0.5, seed=0)
    X_val_subset = X_val[:500]  # Use subset for speed
    y_val_subset = y_val[:500]
    print(f"✓ Loaded {len(X_val_subset)} validation examples")
    print()

    # === Analysis 1: Weight Statistics ===
    print("[3/5] Computing weight statistics...")
    teacher_stats = compute_weight_statistics(teacher)
    logit_stats = compute_weight_statistics(student_logit)
    attention_stats = compute_weight_statistics(student_attention)
    feature_stats = compute_weight_statistics(student_feature)

    print("\n--- Weight Statistics ---")
    print(f"{'Model':<20} {'Params':<12} {'L1 Norm':<12} {'L2 Norm':<12} {'Sparsity':<10}")
    print("-" * 70)
    print(f"{'Teacher (128d)':<20} {teacher_stats['total_params']:<12} {teacher_stats['L1']:<12.2f} {teacher_stats['L2']:<12.2f} {teacher_stats['sparsity']:<10.4f}")
    print(f"{'Student Logit':<20} {logit_stats['total_params']:<12} {logit_stats['L1']:<12.2f} {logit_stats['L2']:<12.2f} {logit_stats['sparsity']:<10.4f}")
    print(f"{'Student Attention':<20} {attention_stats['total_params']:<12} {attention_stats['L1']:<12.2f} {attention_stats['L2']:<12.2f} {attention_stats['sparsity']:<10.4f}")
    print(f"{'Student Feature':<20} {feature_stats['total_params']:<12} {feature_stats['L1']:<12.2f} {feature_stats['L2']:<12.2f} {feature_stats['sparsity']:<10.4f}")
    print()

    # === Analysis 2: CKA and Hidden State Rank ===
    print("[4/5] Computing CKA and hidden state rank...")

    # Forward pass to get intermediate representations
    _, teacher_aux = teacher(X_val_subset, training=False, return_intermediates=True)
    _, logit_aux = student_logit(X_val_subset, training=False, return_intermediates=True)
    _, attention_aux = student_attention(X_val_subset, training=False, return_intermediates=True)
    _, feature_aux = student_feature(X_val_subset, training=False, return_intermediates=True)

    print("\n--- Layer-wise CKA (vs Teacher) ---")
    print(f"{'Layer':<10} {'Logit':<12} {'Attention':<12} {'Feature':<12}")
    print("-" * 50)

    for layer in range(teacher.config.depth):
        # Flatten batch and sequence dimensions
        T = teacher_aux["hidden_states"][layer].reshape(-1, teacher.config.dim)
        L = logit_aux["hidden_states"][layer].reshape(-1, student_logit.config.dim)
        A = attention_aux["hidden_states"][layer].reshape(-1, student_attention.config.dim)
        F = feature_aux["hidden_states"][layer].reshape(-1, student_feature.config.dim)

        cka_logit = centered_kernel_alignment(T, L)
        cka_attention = centered_kernel_alignment(T, A)
        cka_feature = centered_kernel_alignment(T, F)

        print(f"{'Layer ' + str(layer):<10} {cka_logit:<12.4f} {cka_attention:<12.4f} {cka_feature:<12.4f}")

    print("\n--- Hidden State Effective Rank ---")
    print(f"{'Layer':<10} {'Teacher':<12} {'Logit':<12} {'Attention':<12} {'Feature':<12}")
    print("-" * 65)

    for layer in range(teacher.config.depth):
        T = teacher_aux["hidden_states"][layer].reshape(-1, teacher.config.dim)
        L = logit_aux["hidden_states"][layer].reshape(-1, student_logit.config.dim)
        A = attention_aux["hidden_states"][layer].reshape(-1, student_attention.config.dim)
        F = feature_aux["hidden_states"][layer].reshape(-1, student_feature.config.dim)

        rank_teacher = effective_rank(T)
        rank_logit = effective_rank(L)
        rank_attention = effective_rank(A)
        rank_feature = effective_rank(F)

        print(f"{'Layer ' + str(layer):<10} {rank_teacher:<12.2f} {rank_logit:<12.2f} {rank_attention:<12.2f} {rank_feature:<12.2f}")

    print()

    # === Analysis 3: Attention Entropy ===
    print("[5/5] Computing attention entropy...")

    teacher_entropies = [attention_entropy(a) for a in teacher_aux["attentions"]]
    logit_entropies = [attention_entropy(a) for a in logit_aux["attentions"]]
    attention_entropies = [attention_entropy(a) for a in attention_aux["attentions"]]
    feature_entropies = [attention_entropy(a) for a in feature_aux["attentions"]]

    print("\n--- Attention Entropy (bits) ---")
    print(f"{'Layer':<10} {'Teacher':<12} {'Logit':<12} {'Attention':<12} {'Feature':<12}")
    print("-" * 65)

    for layer in range(teacher.config.depth):
        print(f"{'Layer ' + str(layer):<10} {teacher_entropies[layer]:<12.4f} {logit_entropies[layer]:<12.4f} {attention_entropies[layer]:<12.4f} {feature_entropies[layer]:<12.4f}")

    print()
    print("="*70)
    print("✓ Analysis complete!")
    print("="*70)
    print()

    # Save results
    results = {
        "weight_statistics": {
            "teacher": teacher_stats,
            "student_logit": logit_stats,
            "student_attention": attention_stats,
            "student_feature": feature_stats,
        },
        "cka_scores": {
            f"layer_{layer}": {
                "logit": float(centered_kernel_alignment(
                    teacher_aux["hidden_states"][layer].reshape(-1, teacher.config.dim),
                    logit_aux["hidden_states"][layer].reshape(-1, student_logit.config.dim)
                )),
                "attention": float(centered_kernel_alignment(
                    teacher_aux["hidden_states"][layer].reshape(-1, teacher.config.dim),
                    attention_aux["hidden_states"][layer].reshape(-1, student_attention.config.dim)
                )),
                "feature": float(centered_kernel_alignment(
                    teacher_aux["hidden_states"][layer].reshape(-1, teacher.config.dim),
                    feature_aux["hidden_states"][layer].reshape(-1, student_feature.config.dim)
                )),
            }
            for layer in range(teacher.config.depth)
        },
        "effective_ranks": {
            f"layer_{layer}": {
                "teacher": float(effective_rank(teacher_aux["hidden_states"][layer].reshape(-1, teacher.config.dim))),
                "logit": float(effective_rank(logit_aux["hidden_states"][layer].reshape(-1, student_logit.config.dim))),
                "attention": float(effective_rank(attention_aux["hidden_states"][layer].reshape(-1, student_attention.config.dim))),
                "feature": float(effective_rank(feature_aux["hidden_states"][layer].reshape(-1, student_feature.config.dim))),
            }
            for layer in range(teacher.config.depth)
        },
        "attention_entropies": {
            f"layer_{layer}": {
                "teacher": teacher_entropies[layer],
                "logit": logit_entropies[layer],
                "attention": attention_entropies[layer],
                "feature": feature_entropies[layer],
            }
            for layer in range(teacher.config.depth)
        },
    }

    output_file = Path("mechanistic_analysis_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    return results


if __name__ == "__main__":
    analyze_models()
