import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from checkpointing import (
    CheckpointMetadata,
    read_metadata,
    restore_checkpoint,
    save_checkpoint,
)
from models import Transformer, TransformerConfig
from train_nnx import create_optimizer


def test_checkpoint_roundtrip(tmp_path: Path):
    config = TransformerConfig(depth=1, dim=16, heads=1, n_tokens=13, seq_len=4, dropout=0.0)
    rngs = nnx.Rngs(params=0, dropout=0)
    model = Transformer(config, rngs)

    optimizer = create_optimizer(learning_rate=1e-3, warmup_steps=0, beta1=0.9, beta2=0.98, weight_decay=0.0)
    opt_state = optimizer.init(nnx.state(model, nnx.Param))
    rng_key = jax.random.PRNGKey(0)

    # Mutate parameters deterministically
    param_state = nnx.state(model, nnx.Param)
    for leaf in jax.tree.leaves(param_state):
        leaf += 1.0
    nnx.update(model, param_state)

    metadata = CheckpointMetadata(
        config=dataclasses.asdict(config),
        optimizer={
            "learning_rate": 1e-3,
            "beta1": 0.9,
            "beta2": 0.98,
            "weight_decay": 0.0,
            "warmup_steps": 0,
        },
        seed=0,
    )

    save_checkpoint(tmp_path, model, opt_state, rng_key, step=5, epoch=1, metadata=metadata)

    stored_metadata = read_metadata(tmp_path)
    assert stored_metadata is not None
    assert stored_metadata.latest_step == 5

    new_model = Transformer(config, nnx.Rngs(params=0, dropout=0))
    new_opt_state = optimizer.init(nnx.state(new_model, nnx.Param))
    restore_result = restore_checkpoint(tmp_path, new_model, new_opt_state, rng_key, stored_metadata)
    assert restore_result is not None
    assert restore_result.step == 5
    assert restore_result.epoch == 1

    restored_params = nnx.state(new_model, nnx.Param)
    for original, restored in zip(jax.tree.leaves(param_state), jax.tree.leaves(restored_params)):
        assert jnp.allclose(original, restored)

