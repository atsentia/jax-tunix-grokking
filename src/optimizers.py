"""
Optimizer factory for grokking experiments.

Supports:
- AdamW (original optimizer)
- Muon (newer momentum-based optimizer)
"""

import optax
from typing import Optional


def create_adamw_optimizer(
    learning_rate: float,
    warmup_steps: int,
    beta1: float = 0.9,
    beta2: float = 0.98,
    weight_decay: float = 1.0,
    eps: float = 1e-8
) -> optax.GradientTransformation:
    """
    Create AdamW optimizer with linear warmup.

    This is the original optimizer used for grokking experiments.
    Critical hyperparameters for grokking:
    - weight_decay=1.0 (high value essential for grokking phenomenon)
    - Linear warmup over first 10 steps
    - beta2=0.98 (slightly higher than typical 0.999)

    Args:
        learning_rate: Peak learning rate after warmup
        warmup_steps: Number of steps for linear warmup
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        weight_decay: Weight decay coefficient (high for grokking)
        eps: Small constant for numerical stability

    Returns:
        Optax GradientTransformation
    """
    # Learning rate schedule with warmup
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps
    )
    constant_fn = optax.constant_schedule(value=learning_rate)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, constant_fn],
        boundaries=[warmup_steps]
    )

    # AdamW optimizer
    optimizer = optax.adamw(
        learning_rate=schedule_fn,
        b1=beta1,
        b2=beta2,
        eps=eps,
        weight_decay=weight_decay
    )

    return optimizer


def create_muon_optimizer(
    learning_rate: float,
    warmup_steps: int,
    beta: float = 0.95,
    ns_steps: int = 5,
    weight_decay: float = 1.0
) -> optax.GradientTransformation:
    """
    Create Muon optimizer with linear warmup.

    Muon is a momentum-based optimizer that uses Newton-Schulz orthogonalization.
    It's designed to be more efficient than AdamW, especially with larger batch sizes.

    Key features:
    - Works on 2D parameters (matrices) with Newton-Schulz
    - Falls back to Adam for non-2D parameters (embeddings, norms)
    - More data-efficient with large batches

    Args:
        learning_rate: Peak learning rate after warmup
        warmup_steps: Number of steps for linear warmup
        beta: Decay rate for momentum (default: 0.95)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        weight_decay: Weight decay coefficient

    Returns:
        Optax GradientTransformation
    """
    # Learning rate schedule with warmup
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps
    )
    constant_fn = optax.constant_schedule(value=learning_rate)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, constant_fn],
        boundaries=[warmup_steps]
    )

    # Muon optimizer (from optax.contrib)
    try:
        # Muon has weight_decay built-in, so don't add it separately
        optimizer = optax.contrib.muon(
            learning_rate=schedule_fn,
            beta=beta,
            ns_steps=ns_steps,
            weight_decay=weight_decay
        )
    except AttributeError:
        # Fallback if muon not available in current optax version
        print("Warning: optax.contrib.muon not available. Install latest optax from git:")
        print("  pip install git+https://github.com/google-deepmind/optax.git")
        print("Falling back to AdamW...")
        return create_adamw_optimizer(
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay
        )

    return optimizer


def create_optimizer(
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-3,
    warmup_steps: int = 10,
    weight_decay: float = 1.0,
    # AdamW specific
    beta1: float = 0.9,
    beta2: float = 0.98,
    # Muon specific
    beta_muon: float = 0.95,
    ns_steps: int = 5
) -> optax.GradientTransformation:
    """
    Factory function to create optimizer by type.

    Args:
        optimizer_type: Type of optimizer ('adamw' or 'muon')
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        weight_decay: Weight decay coefficient
        beta1: AdamW beta1
        beta2: AdamW beta2
        beta_muon: Muon beta (momentum decay)
        ns_steps: Muon Newton-Schulz steps

    Returns:
        Optax GradientTransformation

    Raises:
        ValueError: If optimizer_type is unknown
    """
    if optimizer_type == 'adamw':
        return create_adamw_optimizer(
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'muon':
        return create_muon_optimizer(
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            beta=beta_muon,
            ns_steps=ns_steps,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose 'adamw' or 'muon'.")


if __name__ == "__main__":
    # Test optimizer creation
    print("Testing optimizer creation...")

    # Test AdamW
    print("\n1. Creating AdamW optimizer...")
    adamw = create_optimizer('adamw', learning_rate=1e-3)
    print("✓ AdamW optimizer created")

    # Test Muon
    print("\n2. Creating Muon optimizer...")
    try:
        muon = create_optimizer('muon', learning_rate=1e-3)
        print("✓ Muon optimizer created")
    except Exception as e:
        print(f"✗ Muon optimizer failed: {e}")

    # Test factory with invalid type
    print("\n3. Testing invalid optimizer type...")
    try:
        invalid = create_optimizer('invalid')
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")

    print("\n✓ Optimizer tests completed!")
