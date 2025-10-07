# models.py - Grokking Transformer in Flax NNX
# Ported from jax_grokking/models.py (Linen) to NNX for Tunix compatibility

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional
import dataclasses


# ============================================================================
# RMSNorm - Root Mean Square Layer Normalization
# ============================================================================

class RMSNorm(nnx.Module):
    """
    Root Mean Square Layer Normalization using Flax NNX.

    Critical for grokking parity:
    - epsilon=1e-6 (must match baseline exactly)
    - use_fast_variance=False for numerical stability (per PRD)
    - Scale parameter initialized to ones
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        use_fast_variance: bool = False,
        rngs: Optional[nnx.Rngs] = None
    ):
        """
        Initialize RMSNorm layer.

        Args:
            dim: Feature dimension
            eps: Epsilon for numerical stability (default: 1e-6 per PRD)
            use_fast_variance: Use fast variance calculation (default: False per PRD)
            rngs: Random number generators (not used, for API compatibility)
        """
        self.dim = dim
        self.eps = eps
        self.use_fast_variance = use_fast_variance

        # Scale parameter (initialized to ones)
        self.scale = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply RMSNorm.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor [..., dim]
        """
        assert x.shape[-1] == self.dim, f"Expected last dim {self.dim}, got {x.shape[-1]}"

        # Compute RMS: sqrt(mean(x^2) + eps)
        if self.use_fast_variance:
            # Fast but potentially less stable
            rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        else:
            # Stable computation (per PRD recommendation)
            variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
            rms = jnp.sqrt(variance + self.eps)

        # Normalize and scale
        normalized = x / rms
        return normalized * self.scale.value


# ============================================================================
# Rotary Positional Embedding (RoPE)
# ============================================================================

def apply_rope(
    x: jax.Array,
    base: float = 1e6
) -> jax.Array:
    """
    Apply Rotary Positional Embeddings (RoPE) to Q or K tensors.

    Critical for grokking parity:
    - base=1e6 (must match baseline exactly, NOT 10000)
    - Applied pre-softmax to queries and keys
    - Rotation based on position indices

    Args:
        x: Input tensor [batch, seq, heads, head_dim]
        base: Base for frequency calculation (default: 1e6 per baseline)

    Returns:
        Tensor with RoPE applied [batch, seq, heads, head_dim]
    """
    b, seq, n_heads, dim = x.shape
    half = dim // 2

    assert half * 2 == dim, f"Head dimension must be even for RoPE, got {dim}"

    # Frequency calculation: theta_i = 1 / (base^(2i/d))
    i = jnp.arange(half)
    theta = 1.0 / (base ** (2 * i / dim))  # Shape: [half]

    # Position indices
    pos = jnp.arange(seq)  # Shape: [seq]
    angles = pos[:, None] * theta[None, :]  # Shape: [seq, half]

    # Compute cos and sin
    cos = jnp.cos(angles)  # Shape: [seq, half]
    sin = jnp.sin(angles)  # Shape: [seq, half]

    # Broadcast to [batch, seq, heads, half]
    cos = jnp.tile(cos[None, :, None, :], (b, 1, n_heads, 1))
    sin = jnp.tile(sin[None, :, None, :], (b, 1, n_heads, 1))

    # Split x into two halves
    x1, x2 = jnp.split(x, 2, axis=-1)  # Each: [b, seq, heads, half]

    # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    return jnp.concatenate([x1_rot, x2_rot], axis=-1)


# ============================================================================
# Multi-Head Self-Attention with Causal Mask
# ============================================================================

class MultiHeadSelfAttention(nnx.Module):
    """
    Multi-Head Self-Attention with causal masking and RoPE.

    Critical for grokking parity:
    - Causal mask prevents future token leakage
    - RoPE applied to Q and K before attention
    - Pre-normalization (RMSNorm before attention)
    - Scaled dot-product attention
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        rngs: Optional[nnx.Rngs] = None
    ):
        """
        Initialize Multi-Head Self-Attention.

        Args:
            dim: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            rngs: Random number generators for parameter initialization
        """
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"

        self.dim = dim
        self.n_heads = n_heads
        self.dim_head = dim // n_heads
        self.dropout_rate = dropout

        # Pre-normalization
        self.norm = RMSNorm(dim, eps=1e-6, use_fast_variance=False)

        # Query, Key, Value projections (no bias per baseline)
        self.Wq = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.Wk = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.Wv = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

        # Output projection
        self.Wo = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)

        # Dropout
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs) if dropout > 0 else None

    def __call__(
        self,
        x: jax.Array,
        training: bool = True,
        return_attention: bool = False,
    ) -> jax.Array:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor [batch, seq, dim]
            training: Whether in training mode (for dropout)
            return_attention: If True, also return the attention weights.

        Returns:
            Output tensor [batch, seq, dim] or tuple(output, attention_weights)
        """
        b, seq, d = x.shape
        assert d == self.dim, f"Expected dim {self.dim}, got {d}"

        # Pre-normalization
        x_norm = self.norm(x)

        # Project to Q, K, V and reshape to [batch, seq, heads, dim_head]
        q = self.Wq(x_norm).reshape(b, seq, self.n_heads, self.dim_head)
        k = self.Wk(x_norm).reshape(b, seq, self.n_heads, self.dim_head)
        v = self.Wv(x_norm).reshape(b, seq, self.n_heads, self.dim_head)

        # Apply RoPE to Q and K
        q = apply_rope(q, base=1e6)
        k = apply_rope(k, base=1e6)

        # Scaled dot-product attention
        # Scores: [batch, heads, seq_q, seq_k]
        attn_scores = jnp.einsum('bthd,bshd->bhts', q, k) / jnp.sqrt(self.dim_head)

        # Causal mask: prevent attention to future tokens
        # Create upper triangular mask with -inf
        causal_mask = jnp.triu(
            jnp.full((seq, seq), -jnp.inf, dtype=jnp.float32),
            k=1
        )  # Shape: [seq, seq]
        causal_mask = causal_mask[None, None, :, :]  # Broadcast to [1, 1, seq, seq]

        # Apply mask
        attn_scores = attn_scores + causal_mask

        # Softmax to get attention weights
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)  # [batch, heads, seq_q, seq_k]

        # Weighted sum of values
        out = jnp.einsum('bhts,bshd->bthd', attn_weights, v)  # [batch, seq, heads, dim_head]
        out = out.reshape(b, seq, self.dim)

        # Output projection
        out = self.Wo(out)

        # Dropout
        if self.dropout is not None and training:
            out = self.dropout(out)

        if return_attention:
            return out, attn_weights

        return out


# ============================================================================
# Feed-Forward Network with Gating (SiLU activation)
# ============================================================================

class FeedForward(nnx.Module):
    """
    Feed-Forward Network with gating mechanism and SiLU activation.

    Critical for grokking parity:
    - SiLU (Swish) activation function
    - Gating mechanism with w1 and w3 branches
    - Pre-normalization (RMSNorm before FFN)
    - Hidden dimension typically 4x model dimension
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        rngs: Optional[nnx.Rngs] = None
    ):
        """
        Initialize Feed-Forward Network.

        Args:
            dim: Model dimension
            hidden_dim: Hidden layer dimension (typically 4 * dim)
            dropout: Dropout rate
            rngs: Random number generators for parameter initialization
        """
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # Pre-normalization
        self.norm = RMSNorm(dim, eps=1e-6, use_fast_variance=False)

        # FFN layers (no bias per baseline)
        self.w1 = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs)  # SiLU branch
        self.w2 = nnx.Linear(hidden_dim, dim, use_bias=False, rngs=rngs)  # Output projection
        self.w3 = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs)  # Gating branch

        # Dropout
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs) if dropout > 0 else None

    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        """
        Apply feed-forward network.

        Args:
            x: Input tensor [batch, seq, dim]
            training: Whether in training mode (for dropout)

        Returns:
            Output tensor [batch, seq, dim]
        """
        assert x.shape[-1] == self.dim, f"Expected dim {self.dim}, got {x.shape[-1]}"

        # Pre-normalization
        x_norm = self.norm(x)

        # FFN with gating: SiLU(w1(x)) * w3(x)
        x1 = self.w1(x_norm)  # [batch, seq, hidden_dim]
        x_silu = jax.nn.silu(x1)  # SiLU activation

        # Gating branch
        x2 = self.w3(x_norm)  # [batch, seq, hidden_dim]
        gated = x_silu * x2

        # Dropout on gated activations
        if self.dropout is not None and training:
            gated = self.dropout(gated)

        # Output projection
        out = self.w2(gated)  # [batch, seq, dim]

        return out


# ============================================================================
# Transformer Model for Grokking
# ============================================================================

@dataclasses.dataclass
class TransformerConfig:
    """Configuration for Grokking Transformer."""
    depth: int = 2              # Number of transformer layers
    dim: int = 128              # Model dimension
    heads: int = 1              # Number of attention heads
    n_tokens: int = 99          # Vocabulary size (p + 2 for op and eq tokens)
    seq_len: int = 4            # Sequence length
    dropout: float = 0.2        # Dropout rate
    pool: str = 'cls'           # Pooling strategy ('cls' or 'mean')


class Transformer(nnx.Module):
    """
    Transformer model for grokking modular arithmetic.

    Critical for grokking parity:
    - Token embeddings initialized with default NNX initializers
    - Residual connections around attention and FFN
    - Pre-normalization architecture (RMSNorm before each layer)
    - Final normalization before output projection
    - Pooling strategy: 'cls' (last token) or 'mean'
    """

    def __init__(
        self,
        config: TransformerConfig,
        rngs: nnx.Rngs
    ):
        """
        Initialize Transformer.

        Args:
            config: Transformer configuration
            rngs: Random number generators for parameter initialization
        """
        self.config = config

        # Token embeddings
        # NNX Embed uses default initializer which should match baseline distribution
        self.embedding = nnx.Embed(
            num_embeddings=config.n_tokens,
            features=config.dim,
            rngs=rngs
        )

        # Transformer blocks: attention + FFN modules stored separately
        self.attention_layers = nnx.List()
        self.ffn_layers = nnx.List()
        for _ in range(config.depth):
            attn = MultiHeadSelfAttention(
                dim=config.dim,
                n_heads=config.heads,
                dropout=config.dropout,
                rngs=rngs
            )
            ffn = FeedForward(
                dim=config.dim,
                hidden_dim=4 * config.dim,  # Standard 4x expansion
                dropout=config.dropout,
                rngs=rngs
            )
            self.attention_layers.append(attn)
            self.ffn_layers.append(ffn)

        # Final normalization
        self.final_norm = RMSNorm(config.dim, eps=1e-6, use_fast_variance=False)

        # Output projection (no bias)
        self.output_dense = nnx.Linear(
            config.dim,
            config.n_tokens,
            use_bias=False,
            rngs=rngs
        )

    def __call__(
        self,
        x: jax.Array,
        training: bool = True,
        return_intermediates: bool = False,
    ) -> jax.Array:
        """
        Forward pass through transformer.

        Args:
            x: Input token indices [batch, seq_len] (int32)
            training: Whether in training mode (for dropout)
            return_intermediates: Whether to return attention maps/hidden states.

        Returns:
            Logits [batch, n_tokens] or tuple(logits, aux)
        """
        assert x.dtype in [jnp.int32, jnp.int64], f"Expected integer input, got {x.dtype}"
        assert x.ndim == 2, f"Expected 2D input [batch, seq], got shape {x.shape}"

        # Token embeddings
        x = self.embedding(x)  # [batch, seq_len, dim]

        attentions = []
        hidden_states = []
        if return_intermediates:
            hidden_states.append(x)

        # Transformer blocks with residual connections
        for attn, ffn in zip(self.attention_layers, self.ffn_layers):
            # Attention with residual
            attn_out = attn(x, training=training, return_attention=return_intermediates)
            if return_intermediates:
                attn_out, attn_weights = attn_out
                attentions.append(attn_weights)
            x = x + attn_out
            if return_intermediates:
                hidden_states.append(x)

            # FFN with residual
            x = x + ffn(x, training=training)
            if return_intermediates:
                hidden_states.append(x)

        # Final normalization
        x = self.final_norm(x)
        if return_intermediates:
            hidden_states.append(x)

        # Pooling
        if self.config.pool == 'mean':
            pooled = jnp.mean(x, axis=1)  # [batch, dim]
        else:  # 'cls' - use last token
            pooled = x[:, -1, :]  # [batch, dim]

        # Output projection
        logits = self.output_dense(pooled)  # [batch, n_tokens]

        if return_intermediates:
            aux = {
                "attentions": attentions,
                "hidden_states": hidden_states,
                "pooled": pooled,
            }
            return logits, aux

        return logits


# ============================================================================
# Helper function for creating model from baseline-style args
# ============================================================================

def create_transformer(
    depth: int,
    dim: int,
    heads: int,
    n_tokens: int,
    seq_len: int,
    dropout: float = 0.2,
    pool: str = 'cls',
    seed: int = 42
) -> Transformer:
    """
    Create a Transformer model (convenience function matching baseline API).

    Args:
        depth: Number of transformer layers
        dim: Model dimension
        heads: Number of attention heads
        n_tokens: Vocabulary size
        seq_len: Sequence length
        dropout: Dropout rate
        pool: Pooling strategy ('cls' or 'mean')
        seed: Random seed for initialization

    Returns:
        Initialized Transformer model
    """
    config = TransformerConfig(
        depth=depth,
        dim=dim,
        heads=heads,
        n_tokens=n_tokens,
        seq_len=seq_len,
        dropout=dropout,
        pool=pool
    )

    rngs = nnx.Rngs(params=seed)
    return Transformer(config, rngs)


if __name__ == "__main__":
    # Quick test
    print("Testing Grokking Transformer (NNX)...")

    # Test configuration
    batch_size = 4
    seq_len = 4
    p = 7
    n_tokens = p + 2  # 0..p-1, op token (p), eq token (p+1)

    # Create model
    config = TransformerConfig(
        depth=2,
        dim=32,
        heads=2,
        n_tokens=n_tokens,
        seq_len=seq_len,
        dropout=0.1,
        pool='cls'
    )

    rngs = nnx.Rngs(params=42, dropout=0)
    model = Transformer(config, rngs)

    # Test forward pass
    x = jnp.array([[1, 7, 3, 8], [2, 7, 4, 8], [0, 7, 5, 8], [3, 7, 6, 8]], dtype=jnp.int32)
    print(f"Input shape: {x.shape}")

    logits = model(x, training=False)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {n_tokens})")

    assert logits.shape == (batch_size, n_tokens), "Output shape mismatch"
    assert jnp.all(jnp.isfinite(logits)), "Output contains non-finite values"

    print("✓ Model test passed!")
