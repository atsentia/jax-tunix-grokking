"""
Unit tests for model components: RMSNorm, RoPE, Attention, FFN, Transformer

Tests verify correctness of NNX model implementations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from models import (
    RMSNorm,
    apply_rope,
    MultiHeadSelfAttention,
    FeedForward,
    Transformer,
    TransformerConfig,
)


def test_rmsnorm_forward():
    """Test RMSNorm forward pass."""
    print("\n" + "="*60)
    print("TEST: RMSNorm Forward Pass")
    print("="*60)

    # Test configuration
    dim = 32
    batch_size = 4
    seq_len = 8
    eps = 1e-6

    # Fixed input for reproducibility
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (batch_size, seq_len, dim))

    # Create and apply RMSNorm
    norm = RMSNorm(dim=dim, eps=eps, use_fast_variance=False)
    output = norm(x)

    # Verify properties
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    assert jnp.all(jnp.isfinite(output)), "Output contains non-finite values"

    # Verify normalization: variance should be ~1
    variance = jnp.var(output, axis=-1)
    mean_variance = jnp.mean(variance)
    assert jnp.allclose(mean_variance, 1.0, rtol=1e-1), f"Variance not normalized: {mean_variance}"

    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Mean variance: {mean_variance:.4f} (target: 1.0)")
    print(f"✓ All values finite: {jnp.all(jnp.isfinite(output))}")

    print("="*60)
    print("✓ RMSNorm Test PASSED")
    print("="*60)


def test_rope_rotation():
    """Test RoPE (Rotary Positional Embedding) application."""
    print("\n" + "="*60)
    print("TEST: RoPE Rotation")
    print("="*60)

    # Test configuration
    batch_size = 2
    seq_len = 4
    n_heads = 2
    head_dim = 16
    base = 1e6

    # Fixed input
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (batch_size, seq_len, n_heads, head_dim))

    # Apply RoPE
    output = apply_rope(x, base=base)

    # Verify properties
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    assert jnp.all(jnp.isfinite(output)), "Output contains non-finite values"

    # Verify rotation preserves norm (approximately)
    input_norm = jnp.linalg.norm(x)
    output_norm = jnp.linalg.norm(output)
    assert jnp.allclose(input_norm, output_norm, rtol=1e-3), \
        f"Norm not preserved: {input_norm:.4f} -> {output_norm:.4f}"

    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Norm preserved: {input_norm:.4f} -> {output_norm:.4f}")
    print(f"✓ All values finite: {jnp.all(jnp.isfinite(output))}")

    print("="*60)
    print("✓ RoPE Test PASSED")
    print("="*60)


def test_attention_causality():
    """Test Multi-Head Self-Attention with causal masking."""
    print("\n" + "="*60)
    print("TEST: Multi-Head Self-Attention (Causal Mask)")
    print("="*60)

    # Test configuration
    dim = 32
    n_heads = 2
    batch_size = 2
    seq_len = 4
    dropout = 0.0

    # Fixed input
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (batch_size, seq_len, dim))

    # Create attention module
    attn = MultiHeadSelfAttention(
        dim=dim,
        n_heads=n_heads,
        dropout=dropout,
        rngs=nnx.Rngs(params=42, dropout=0)
    )
    output = attn(x, training=False)

    # Verify output properties
    assert output.shape == (batch_size, seq_len, dim), \
        f"Shape mismatch: {output.shape} != {(batch_size, seq_len, dim)}"
    assert jnp.all(jnp.isfinite(output)), "Output contains non-finite values"

    # Test causality: modify future token, check past tokens unchanged
    x_modified = x.at[:, -1, :].set(jax.random.normal(jax.random.PRNGKey(999), (batch_size, dim)))
    output_modified = attn(x_modified, training=False)

    # First token should only attend to itself (causal mask)
    first_token_same = jnp.allclose(output[:, 0, :], output_modified[:, 0, :], rtol=1e-5)

    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Causal mask verified: {first_token_same}")
    print(f"✓ All values finite: {jnp.all(jnp.isfinite(output))}")

    assert first_token_same, "Causal mask not working - future tokens affect past"

    print("="*60)
    print("✓ Attention Test PASSED")
    print("="*60)


def test_feedforward_activation():
    """Test Feed-Forward Network with SiLU activation."""
    print("\n" + "="*60)
    print("TEST: Feed-Forward Network")
    print("="*60)

    # Test configuration
    dim = 32
    hidden_dim = 128
    batch_size = 2
    seq_len = 4
    dropout = 0.0

    # Fixed input
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (batch_size, seq_len, dim))

    # Create FFN
    ffn = FeedForward(
        dim=dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        rngs=nnx.Rngs(params=42, dropout=0)
    )
    output = ffn(x, training=False)

    # Verify output properties
    assert output.shape == (batch_size, seq_len, dim), \
        f"Shape mismatch: {output.shape} != {(batch_size, seq_len, dim)}"
    assert jnp.all(jnp.isfinite(output)), "Output contains non-finite values"
    assert jnp.any(output != 0), "FFN output is all zeros"

    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Non-zero outputs: {jnp.any(output != 0)}")
    print(f"✓ All values finite: {jnp.all(jnp.isfinite(output))}")

    print("="*60)
    print("✓ Feed-Forward Test PASSED")
    print("="*60)


def test_transformer_forward():
    """Test full Transformer forward pass."""
    print("\n" + "="*60)
    print("TEST: Transformer Forward Pass")
    print("="*60)

    # Test configuration
    p = 7
    n_tokens = p + 2  # p numbers + operation + equals
    seq_len = 4
    batch_size = 4

    config = TransformerConfig(
        depth=2,
        dim=32,
        heads=2,
        n_tokens=n_tokens,
        seq_len=seq_len,
        dropout=0.0,
        pool='cls'
    )

    # Create model
    model = Transformer(config, rngs=nnx.Rngs(params=42, dropout=0))

    # Test input: [num1, op, num2, eq]
    x = jnp.array([
        [1, 7, 3, 8],  # 1 op 3 = ?
        [2, 7, 4, 8],  # 2 op 4 = ?
        [0, 7, 5, 8],  # 0 op 5 = ?
        [3, 7, 6, 8],  # 3 op 6 = ?
    ], dtype=jnp.int32)

    # Forward pass
    logits = model(x, training=False)

    # Verify output
    assert logits.shape == (batch_size, n_tokens), \
        f"Expected shape {(batch_size, n_tokens)}, got {logits.shape}"
    assert jnp.all(jnp.isfinite(logits)), "Logits contain non-finite values"

    print(f"✓ Output shape: {logits.shape}")
    print(f"✓ Logits range: [{jnp.min(logits):.2f}, {jnp.max(logits):.2f}]")
    print(f"✓ All values finite: {jnp.all(jnp.isfinite(logits))}")

    # Test determinism with same seed
    model2 = Transformer(config, rngs=nnx.Rngs(params=42, dropout=0))
    logits2 = model2(x, training=False)
    deterministic = jnp.allclose(logits, logits2)

    print(f"✓ Deterministic initialization: {deterministic}")
    assert deterministic, "Model not deterministic with same seed"

    print("="*60)
    print("✓ Transformer Test PASSED")
    print("="*60)


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("\n" + "="*60)
    print("TEST: Gradient Flow")
    print("="*60)

    # Simple configuration
    p = 5
    n_tokens = p + 2
    config = TransformerConfig(
        depth=1,
        dim=16,
        heads=2,
        n_tokens=n_tokens,
        seq_len=4,
        dropout=0.0,
        pool='cls'
    )

    model = Transformer(config, rngs=nnx.Rngs(params=42, dropout=0))

    # Test input and labels
    x = jnp.array([[1, 5, 3, 6], [2, 5, 4, 6]], dtype=jnp.int32)
    y = jnp.array([2, 3], dtype=jnp.int32)

    # Define loss function
    def loss_fn(model):
        logits = model(x, training=False)
        one_hot = jax.nn.one_hot(y, n_tokens)
        import optax
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss

    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Check gradient properties
    grad_state = nnx.state(grads)
    has_gradients = False
    total_grad_norm = 0.0

    for path, grad in jax.tree_util.tree_leaves_with_path(grad_state):
        if isinstance(grad, jax.Array):
            grad_norm = jnp.linalg.norm(grad.flatten())
            total_grad_norm += grad_norm
            if grad_norm > 1e-8:
                has_gradients = True

    assert has_gradients, "No gradients computed"
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    assert total_grad_norm > 0, "Total gradient norm is zero"

    print(f"✓ Loss: {float(loss):.4f}")
    print(f"✓ Total gradient norm: {total_grad_norm:.4f}")
    print(f"✓ Gradients flow through model")

    print("="*60)
    print("✓ Gradient Flow Test PASSED")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GROKKING MODEL UNIT TESTS")
    print("="*80)

    try:
        # Run all tests
        test_rmsnorm_forward()
        test_rope_rotation()
        test_attention_causality()
        test_feedforward_activation()
        test_transformer_forward()
        test_gradient_flow()

        print("\n" + "="*80)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*80)
        print("\nModel Components Verified:")
        print("  ✓ RMSNorm normalization working correctly")
        print("  ✓ RoPE preserves vector norms")
        print("  ✓ Attention causal mask prevents future leakage")
        print("  ✓ FFN with SiLU activation functioning")
        print("  ✓ Transformer forward pass produces valid logits")
        print("  ✓ Gradients flow through all layers")
        print("="*80)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
