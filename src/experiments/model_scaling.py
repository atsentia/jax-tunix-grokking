"""
Model scaling utilities for creating variants of the grokking transformer.

Supports multiple scaling strategies:
- Width scaling: Change embedding dimension
- Depth scaling: Change number of layers
- Hybrid scaling: Adjust both depth and width
"""

from typing import Tuple, Dict, Any
from dataclasses import asdict
import jax

from flax import nnx
from models import Transformer, TransformerConfig


def count_parameters(model: Transformer) -> int:
    """
    Count total trainable parameters in a model.

    Args:
        model: Transformer model

    Returns:
        Total parameter count
    """
    param_state = nnx.state(model, nnx.Param)
    total = sum(p.size for p in jax.tree.leaves(param_state) if hasattr(p, 'size'))
    return total


def scale_width(base_dim: int, scale_factor: float) -> int:
    """
    Scale width (embedding dimension) by a factor.

    Ensures result is divisible by number of heads and is even (for RoPE).

    Args:
        base_dim: Base embedding dimension
        scale_factor: Scaling factor (e.g., 0.5, 2.0, 5.0)

    Returns:
        Scaled dimension (rounded to nearest valid value)
    """
    scaled = int(base_dim * scale_factor)

    # Ensure even number (required for RoPE)
    if scaled % 2 != 0:
        scaled += 1

    # Minimum dimension
    scaled = max(scaled, 16)

    return scaled


def scale_depth(base_depth: int, scale_factor: float) -> int:
    """
    Scale depth (number of layers) by a factor.

    Args:
        base_depth: Base number of layers
        scale_factor: Scaling factor (e.g., 0.5, 2.0, 5.0)

    Returns:
        Scaled depth (rounded to nearest integer, minimum 1)
    """
    scaled = max(1, round(base_depth * scale_factor))
    return scaled


def create_scaled_config(
    base_config: TransformerConfig,
    scale_factor: float,
    strategy: str = 'width'
) -> TransformerConfig:
    """
    Create a scaled model configuration.

    Args:
        base_config: Base model configuration
        scale_factor: Scaling factor (0.5 = shrink by half, 2.0 = double size)
        strategy: Scaling strategy - 'width', 'depth', or 'hybrid'

    Returns:
        New scaled configuration
    """
    if strategy == 'width':
        # Scale only embedding dimension
        new_dim = scale_width(base_config.dim, scale_factor)
        return TransformerConfig(
            depth=base_config.depth,
            dim=new_dim,
            heads=base_config.heads,
            n_tokens=base_config.n_tokens,
            seq_len=base_config.seq_len,
            dropout=base_config.dropout,
            pool=base_config.pool
        )

    elif strategy == 'depth':
        # Scale only number of layers
        new_depth = scale_depth(base_config.depth, scale_factor)
        return TransformerConfig(
            depth=new_depth,
            dim=base_config.dim,
            heads=base_config.heads,
            n_tokens=base_config.n_tokens,
            seq_len=base_config.seq_len,
            dropout=base_config.dropout,
            pool=base_config.pool
        )

    elif strategy == 'hybrid':
        # Scale both dimensions (geometric mean approach)
        # Total params scale roughly as: depth * dim^2
        # For target scale_factor, we want: new_depth * new_dim^2 ≈ scale_factor * depth * dim^2

        # Use sqrt for balanced scaling
        dim_factor = scale_factor ** 0.5
        depth_factor = scale_factor ** 0.5

        new_dim = scale_width(base_config.dim, dim_factor)
        new_depth = scale_depth(base_config.depth, depth_factor)

        return TransformerConfig(
            depth=new_depth,
            dim=new_dim,
            heads=base_config.heads,
            n_tokens=base_config.n_tokens,
            seq_len=base_config.seq_len,
            dropout=base_config.dropout,
            pool=base_config.pool
        )

    else:
        raise ValueError(f"Unknown scaling strategy: {strategy}")


def create_scaled_model(
    base_config: TransformerConfig,
    scale_factor: float,
    strategy: str = 'width',
    rngs: nnx.Rngs = None
) -> Tuple[Transformer, TransformerConfig]:
    """
    Create a scaled model instance.

    Args:
        base_config: Base model configuration
        scale_factor: Scaling factor
        strategy: Scaling strategy
        rngs: Random number generators (if None, uses default)

    Returns:
        Tuple of (scaled_model, scaled_config)
    """
    scaled_config = create_scaled_config(base_config, scale_factor, strategy)

    if rngs is None:
        rngs = nnx.Rngs(params=42, dropout=0)

    scaled_model = Transformer(scaled_config, rngs)

    return scaled_model, scaled_config


def get_scaling_info(
    base_config: TransformerConfig,
    scaled_config: TransformerConfig,
    base_params: int,
    scaled_params: int
) -> Dict[str, Any]:
    """
    Get detailed information about scaling.

    Args:
        base_config: Base configuration
        scaled_config: Scaled configuration
        base_params: Base parameter count
        scaled_params: Scaled parameter count

    Returns:
        Dictionary with scaling information
    """
    actual_scale = scaled_params / base_params

    return {
        'base': {
            'depth': base_config.depth,
            'dim': base_config.dim,
            'heads': base_config.heads,
            'params': base_params
        },
        'scaled': {
            'depth': scaled_config.depth,
            'dim': scaled_config.dim,
            'heads': scaled_config.heads,
            'params': scaled_params
        },
        'scaling': {
            'depth_ratio': scaled_config.depth / base_config.depth,
            'dim_ratio': scaled_config.dim / base_config.dim,
            'param_ratio': actual_scale
        }
    }


def are_configs_compatible(
    teacher_config: TransformerConfig,
    student_config: TransformerConfig
) -> Tuple[bool, str]:
    """
    Check if two model configurations are compatible for distillation.

    Args:
        teacher_config: Teacher model configuration
        student_config: Student model configuration

    Returns:
        Tuple of (is_compatible, reason)
    """
    # Must have same vocabulary and sequence length
    if teacher_config.n_tokens != student_config.n_tokens:
        return False, f"Token mismatch: teacher={teacher_config.n_tokens}, student={student_config.n_tokens}"

    if teacher_config.seq_len != student_config.seq_len:
        return False, f"Sequence length mismatch: teacher={teacher_config.seq_len}, student={student_config.seq_len}"

    # Must use same pooling strategy
    if teacher_config.pool != student_config.pool:
        return False, f"Pooling mismatch: teacher={teacher_config.pool}, student={student_config.pool}"

    # Student should typically be smaller (though not required)
    teacher_params = estimate_params(teacher_config)
    student_params = estimate_params(student_config)

    if student_params > teacher_params:
        return True, f"Warning: student ({student_params:,}) larger than teacher ({teacher_params:,})"

    return True, "Compatible"


def estimate_params(config: TransformerConfig) -> int:
    """
    Estimate parameter count from configuration (without creating model).

    Approximate formula:
    - Embeddings: n_tokens * dim
    - Per layer: ~4 * dim^2 (attention) + ~12 * dim^2 (FFN with 4x expansion)
    - Output: dim * n_tokens

    Args:
        config: Model configuration

    Returns:
        Estimated parameter count
    """
    # Token embeddings
    embedding_params = config.n_tokens * config.dim

    # Per transformer layer
    # Attention: Wq, Wk, Wv, Wo = 4 * (dim * dim) = 4 * dim^2
    attn_params = 4 * (config.dim ** 2)

    # FFN: w1, w2, w3 with 4x expansion = (dim * 4*dim) + (4*dim * dim) + (dim * 4*dim) = 12 * dim^2
    ffn_params = 12 * (config.dim ** 2)

    # RMSNorm scales (2 per layer + 1 final) ≈ (2 * depth + 1) * dim
    norm_params = (2 * config.depth + 1) * config.dim

    layer_params = (attn_params + ffn_params) * config.depth

    # Output projection
    output_params = config.dim * config.n_tokens

    total = embedding_params + layer_params + norm_params + output_params

    return total


def print_model_comparison(
    base_config: TransformerConfig,
    scaled_config: TransformerConfig,
    base_model: Transformer = None,
    scaled_model: Transformer = None
):
    """
    Print a comparison of two model configurations.

    Args:
        base_config: Base configuration
        scaled_config: Scaled configuration
        base_model: Base model (optional, for exact param count)
        scaled_model: Scaled model (optional, for exact param count)
    """
    if base_model:
        base_params = count_parameters(base_model)
    else:
        base_params = estimate_params(base_config)

    if scaled_model:
        scaled_params = count_parameters(scaled_model)
    else:
        scaled_params = estimate_params(scaled_config)

    info = get_scaling_info(base_config, scaled_config, base_params, scaled_params)

    print("=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(f"\nBase Model:")
    print(f"  Depth: {info['base']['depth']} layers")
    print(f"  Dim: {info['base']['dim']}")
    print(f"  Heads: {info['base']['heads']}")
    print(f"  Params: {info['base']['params']:,}")

    print(f"\nScaled Model:")
    print(f"  Depth: {info['scaled']['depth']} layers")
    print(f"  Dim: {info['scaled']['dim']}")
    print(f"  Heads: {info['scaled']['heads']}")
    print(f"  Params: {info['scaled']['params']:,}")

    print(f"\nScaling Ratios:")
    print(f"  Depth: {info['scaling']['depth_ratio']:.2f}x")
    print(f"  Width: {info['scaling']['dim_ratio']:.2f}x")
    print(f"  Params: {info['scaling']['param_ratio']:.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    # Test scaling utilities
    print("Testing model scaling utilities...")

    # Base configuration (standard grokking model)
    base_config = TransformerConfig(
        depth=2,
        dim=128,
        heads=1,
        n_tokens=99,
        seq_len=4,
        dropout=0.2,
        pool='cls'
    )

    rngs = nnx.Rngs(params=42, dropout=0)
    base_model = Transformer(base_config, rngs)
    base_params = count_parameters(base_model)

    print(f"\nBase model: {base_params:,} parameters\n")

    # Test different scaling strategies
    for strategy in ['width', 'depth', 'hybrid']:
        print(f"\n{'='*80}")
        print(f"Testing {strategy.upper()} scaling")
        print(f"{'='*80}")

        for scale_factor in [0.5, 2.0, 5.0]:
            print(f"\n--- Scale factor: {scale_factor}x ---")

            scaled_model, scaled_config = create_scaled_model(
                base_config, scale_factor, strategy
            )

            print_model_comparison(
                base_config, scaled_config,
                base_model, scaled_model
            )

            # Check compatibility
            compatible, reason = are_configs_compatible(base_config, scaled_config)
            print(f"\nCompatibility: {compatible} - {reason}")

    print("\n✓ Model scaling test completed!")
