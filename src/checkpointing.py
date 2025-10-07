"""Checkpoint utilities built on Orbax for grokking experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx


@dataclass
class CheckpointMetadata:
    """Metadata persisted alongside checkpoints."""

    config: Dict[str, Any]
    optimizer: Dict[str, Any]
    seed: int
    latest_step: Optional[int] = None
    latest_epoch: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "config": self.config,
            "optimizer": self.optimizer,
            "seed": int(self.seed),
            "latest_step": None if self.latest_step is None else int(self.latest_step),
            "latest_epoch": None if self.latest_epoch is None else int(self.latest_epoch),
        }
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CheckpointMetadata":
        return cls(
            config=payload.get("config", {}),
            optimizer=payload.get("optimizer", {}),
            seed=int(payload["seed"]),
            latest_step=payload.get("latest_step"),
            latest_epoch=payload.get("latest_epoch"),
        )


@dataclass
class CheckpointLoadResult:
    """Result returned when restoring a checkpoint."""

    model_state: nnx.State
    optimizer_state: Any
    rng_key: jax.Array
    step: int
    epoch: int
    metadata: CheckpointMetadata


def _metadata_file(directory: Path) -> Path:
    return directory / "metadata.json"


def read_metadata(directory: Path) -> Optional[CheckpointMetadata]:
    """Read checkpoint metadata if it exists."""

    directory = Path(directory)
    file_path = _metadata_file(directory)
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return CheckpointMetadata.from_dict(payload)


def write_metadata(directory: Path, metadata: CheckpointMetadata) -> None:
    """Persist metadata JSON."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = _metadata_file(directory)
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(metadata.to_dict(), fh, indent=2, sort_keys=True)


def _step_directory(directory: Path, step: int) -> Path:
    return directory / f"step_{step:08d}"


def save_checkpoint(
    directory: Path,
    model: nnx.Module,
    optimizer_state: Any,
    rng_key: jax.Array,
    step: int,
    epoch: int,
    metadata: CheckpointMetadata,
) -> None:
    """Save a training checkpoint using Orbax."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": nnx.state(model).to_pure_dict(),
        "optimizer_state": optimizer_state,
        "rng_key": rng_key,
        "step": np.asarray(step, dtype=np.int32),
        "epoch": np.asarray(epoch, dtype=np.int32),
    }

    checkpointer = ocp.PyTreeCheckpointer()
    step_dir = _step_directory(directory, step)
    checkpointer.save(os.fspath(step_dir), payload)

    metadata.latest_step = step
    metadata.latest_epoch = epoch
    write_metadata(directory, metadata)


def restore_checkpoint(
    directory: Path,
    model: nnx.Module,
    optimizer_state_template: Any,
    rng_key_template: jax.Array,
    metadata: CheckpointMetadata,
) -> Optional[CheckpointLoadResult]:
    """Restore the latest checkpoint into the provided model.

    Args:
        directory: Root directory containing checkpoints.
        model: Model instance whose state will be updated.
        optimizer_state_template: Optimizer state with the correct structure.
        rng_key_template: PRNGKey used as template for restoration.
        metadata: Metadata describing latest step/epoch.

    Returns:
        CheckpointLoadResult or None if no checkpoint is available.
    """

    if metadata.latest_step is None:
        return None

    directory = Path(directory)
    step_dir = _step_directory(directory, metadata.latest_step)
    if not step_dir.exists():
        return None

    payload_template = {
        "model_state": nnx.state(model),
        "optimizer_state": optimizer_state_template,
        "rng_key": rng_key_template,
        "step": np.asarray(0, dtype=np.int32),
        "epoch": np.asarray(0, dtype=np.int32),
    }

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(os.fspath(step_dir))

    restored_model_state = restored["model_state"]
    if isinstance(restored_model_state, dict):
        current_state = nnx.state(model)
        current_state.replace_by_pure_dict(restored_model_state)
        nnx.update(model, current_state)
        model_state = current_state
    else:
        nnx.update(model, restored_model_state)
        model_state = restored_model_state

    return CheckpointLoadResult(
        model_state=model_state,
        optimizer_state=restored["optimizer_state"],
        rng_key=restored["rng_key"],
        step=int(restored["step"]),
        epoch=int(restored["epoch"]),
        metadata=metadata,
    )


def load_history(history_path: Path) -> Dict[str, list]:
    """Load training history JSON if available."""

    history_path = Path(history_path)
    if not history_path.exists():
        return {}

    with open(history_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_history(history_path: Path, history: Dict[str, list]) -> None:
    """Write training history to disk."""

    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, sort_keys=True)

