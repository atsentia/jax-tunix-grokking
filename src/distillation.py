"""Knowledge distillation utilities for grokking models using Tunix."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from checkpointing import (
    CheckpointMetadata,
    read_metadata,
    restore_checkpoint,
    save_checkpoint,
    save_history,
)
from models import Transformer, TransformerConfig
from train_nnx import create_optimizer
from data import grokking_data


class DistillationStrategy(str, Enum):
    """Enumeration of supported distillation strategies."""

    LOGIT = "logit"
    ATTENTION = "attention_transfer"
    FEATURE = "feature_projection"


@dataclass
class DataConfig:
    """Dataset configuration used by distillation."""

    p: int = 97
    operation: str = "/"
    train_fraction: float = 0.5
    batch_size: int = 512


@dataclass
class TrainingConfig:
    """Optimization hyper-parameters for student training."""

    epochs: int = 50
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.98
    warmup_steps: int = 0
    optimizer_type: str = "adamw"
    log_every: int = 1


@dataclass
class DistillationConfig:
    """Configuration of the distillation objectives."""

    strategies: Tuple[DistillationStrategy, ...] = (DistillationStrategy.LOGIT,)
    temperature: float = 2.0
    alpha: float = 0.5
    attention_weight: float = 1.0
    feature_weight: float = 1.0
    use_ground_truth: bool = True

    def requires_intermediates(self) -> bool:
        return any(
            strategy in (DistillationStrategy.ATTENTION, DistillationStrategy.FEATURE)
            for strategy in self.strategies
        )

    def requires_attention(self) -> bool:
        return DistillationStrategy.ATTENTION in self.strategies

    def requires_feature_projection(self) -> bool:
        return DistillationStrategy.FEATURE in self.strategies


@dataclass
class DistillationResult:
    """Summary of a completed distillation run."""

    history: Dict[str, List[float]]
    final_metrics: Dict[str, float]
    checkpoint_dir: Optional[Path]
    student_config: TransformerConfig


class TunixLogger:
    """Light-weight adapter around optional Tunix logging APIs."""

    def __init__(self, project: str, run_name: str, enabled: bool = True) -> None:
        self._enabled = enabled
        self._logger = None
        if not enabled:
            return
        try:
            import tunix  # type: ignore

            create_run = getattr(tunix, "create_run", None)
            if callable(create_run):
                self._logger = create_run(name=run_name, project=project)
            else:
                logging_mod = getattr(tunix, "logging", None)
                if logging_mod is not None:
                    run_cls = getattr(logging_mod, "Run", None)
                    if run_cls is not None:
                        self._logger = run_cls(name=run_name, metadata={"project": project})
        except Exception as exc:  # pragma: no cover - best effort integration
            print(f"[Tunix] Logging disabled ({exc})")
            self._logger = None

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        if not self._logger:
            return
        log_fn = getattr(self._logger, "log", None)
        if callable(log_fn):
            try:
                log_fn(step=step, metrics=metrics)
            except Exception as exc:  # pragma: no cover
                print(f"[Tunix] Failed to log metrics: {exc}")


class FeatureProjector(nnx.Module):
    """Linear projectors that map teacher hidden states to student width."""

    def __init__(self, teacher_dim: int, student_dim: int, num_features: int, rngs: nnx.Rngs):
        self.layers = []
        for _ in range(num_features):
            self.layers.append(
                nnx.Linear(teacher_dim, student_dim, use_bias=False, rngs=rngs)
            )

    def __call__(self, features: Sequence[jax.Array]) -> List[jax.Array]:
        projected: List[jax.Array] = []
        for linear, feature in zip(self.layers, features):
            projected.append(linear(feature))
        return projected


class DistillationContainer(nnx.Module):
    """Bundle student model with optional feature projector for optimization."""

    def __init__(self, student: Transformer, projector: Optional[FeatureProjector]):
        self.student = student
        self.projector = projector


def _logit_distillation_loss(
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    temperature: float,
) -> jax.Array:
    scaled_student = student_logits / temperature
    scaled_teacher = teacher_logits / temperature
    student_log_probs = jax.nn.log_softmax(scaled_student, axis=-1)
    teacher_probs = jax.nn.softmax(scaled_teacher, axis=-1)
    kl = jnp.sum(
        teacher_probs * (jnp.log(jnp.clip(teacher_probs, 1e-8, 1.0)) - student_log_probs),
        axis=-1,
    )
    return kl.mean() * (temperature ** 2)


def _attention_transfer_loss(
    student_attn: Sequence[jax.Array],
    teacher_attn: Sequence[jax.Array],
) -> jax.Array:
    losses = []
    for s, t in zip(student_attn, teacher_attn):
        losses.append(jnp.mean((s - t) ** 2))
    if not losses:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.stack(losses).mean()


def _feature_projection_loss(
    student_hidden: Sequence[jax.Array],
    teacher_hidden: Sequence[jax.Array],
) -> jax.Array:
    losses = []
    for s, t in zip(student_hidden, teacher_hidden):
        losses.append(jnp.mean((s - t) ** 2))
    if not losses:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.stack(losses).mean()


def scale_transformer_config(
    base: TransformerConfig,
    scale: float,
    depth_scale: float = 1.0,
) -> TransformerConfig:
    """Scale transformer width/depth while keeping heads and sequence fixed."""

    dim = max(8, int(round(base.dim * scale)))
    if dim % base.heads != 0:
        dim += base.heads - (dim % base.heads)
    depth = max(1, int(round(base.depth * depth_scale)))
    return TransformerConfig(
        depth=depth,
        dim=dim,
        heads=base.heads,
        n_tokens=base.n_tokens,
        seq_len=base.seq_len,
        dropout=base.dropout,
        pool=base.pool,
    )


def _create_history_dict() -> Dict[str, List[float]]:
    history = {
        "step": [],
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "logit_loss": [],
        "attention_loss": [],
        "feature_loss": [],
    }
    return history


def _append_history(
    history: Dict[str, List[float]],
    step: int,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    history["step"].append(step)
    history["epoch"].append(epoch)
    history["train_loss"].append(float(train_metrics.get("loss", 0.0)))
    history["train_acc"].append(float(train_metrics.get("accuracy", 0.0)))
    history["val_loss"].append(float(val_metrics.get("loss", 0.0)))
    history["val_acc"].append(float(val_metrics.get("accuracy", 0.0)))
    history["logit_loss"].append(float(train_metrics.get("logit_loss", 0.0)))
    history["attention_loss"].append(float(train_metrics.get("attention_loss", 0.0)))
    history["feature_loss"].append(float(train_metrics.get("feature_loss", 0.0)))


def _evaluate_student(
    container: DistillationContainer,
    teacher: Optional[Transformer],
    X: jax.Array,
    y: jax.Array,
    n_tokens: int,
    config: DistillationConfig,
) -> Dict[str, float]:
    needs_aux = config.requires_intermediates()
    student_outputs = container.student(
        jnp.asarray(X),
        training=False,
        return_intermediates=needs_aux,
    )
    if needs_aux:
        student_logits, student_aux = student_outputs
    else:
        student_logits = student_outputs
        student_aux = {}

    labels = jnp.asarray(y)
    one_hot = jax.nn.one_hot(labels, n_tokens)
    ce_loss = optax.softmax_cross_entropy(student_logits, one_hot).mean()
    preds = jnp.argmax(student_logits, axis=-1)
    accuracy = jnp.mean((preds == labels).astype(jnp.float32))

    metrics: Dict[str, float] = {
        "loss": float(ce_loss),
        "accuracy": float(accuracy),
        "logit_loss": 0.0,
        "attention_loss": 0.0,
        "feature_loss": 0.0,
    }

    if needs_aux and teacher is not None:
        teacher_outputs = teacher(
            jnp.asarray(X),
            training=False,
            return_intermediates=True,
        )
        teacher_logits, teacher_aux = teacher_outputs
        teacher_logits = jax.lax.stop_gradient(teacher_logits)

        if DistillationStrategy.LOGIT in config.strategies:
            metrics["logit_loss"] = float(
                _logit_distillation_loss(student_logits, teacher_logits, config.temperature)
            )
        if config.requires_attention():
            metrics["attention_loss"] = float(
                _attention_transfer_loss(
                    student_aux.get("attentions", []),
                    [jax.lax.stop_gradient(a) for a in teacher_aux.get("attentions", [])],
                )
            )
        if config.requires_feature_projection():
            teacher_hidden = [jax.lax.stop_gradient(h) for h in teacher_aux.get("hidden_states", [])]
            if container.projector is not None:
                projected_teacher = container.projector(teacher_hidden)
            else:
                projected_teacher = teacher_hidden
            metrics["feature_loss"] = float(
                _feature_projection_loss(
                    student_aux.get("hidden_states", []),
                    projected_teacher,
                )
            )

    return metrics


def _distillation_train_step(
    container: DistillationContainer,
    teacher: Transformer,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch_X: jax.Array,
    batch_y: jax.Array,
    n_tokens: int,
    config: DistillationConfig,
) -> Tuple[DistillationContainer, optax.OptState, Dict[str, float]]:
    needs_aux = config.requires_intermediates()
    batch_X = jnp.asarray(batch_X)
    batch_y = jnp.asarray(batch_y)

    def loss_fn(module: DistillationContainer) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        student_outputs = module.student(
            batch_X,
            training=True,
            return_intermediates=needs_aux,
        )
        if needs_aux:
            student_logits, student_aux = student_outputs
        else:
            student_logits = student_outputs
            student_aux = {}

        teacher_outputs = teacher(
            batch_X,
            training=False,
            return_intermediates=needs_aux,
        )
        if needs_aux:
            teacher_logits, teacher_aux = teacher_outputs
        else:
            teacher_logits = teacher_outputs
            teacher_aux = {}

        teacher_logits = jax.lax.stop_gradient(teacher_logits)
        total_loss = 0.0
        metrics: Dict[str, jax.Array] = {}

        if config.use_ground_truth:
            one_hot = jax.nn.one_hot(batch_y, n_tokens)
            ce_loss = optax.softmax_cross_entropy(student_logits, one_hot).mean()
            total_loss += config.alpha * ce_loss
            metrics["hard_ce"] = ce_loss
        else:
            metrics["hard_ce"] = jnp.array(0.0, dtype=jnp.float32)

        if DistillationStrategy.LOGIT in config.strategies:
            logit_loss = _logit_distillation_loss(student_logits, teacher_logits, config.temperature)
            total_loss += (1.0 - config.alpha) * logit_loss
            metrics["logit_loss"] = logit_loss
        else:
            metrics["logit_loss"] = jnp.array(0.0, dtype=jnp.float32)

        if needs_aux:
            student_attn = student_aux.get("attentions", [])
            student_hidden = student_aux.get("hidden_states", [])
            teacher_attn = [jax.lax.stop_gradient(a) for a in teacher_aux.get("attentions", [])]
            teacher_hidden = [jax.lax.stop_gradient(h) for h in teacher_aux.get("hidden_states", [])]
        else:
            student_attn = []
            student_hidden = []
            teacher_attn = []
            teacher_hidden = []

        if config.requires_attention():
            attention_loss = _attention_transfer_loss(student_attn, teacher_attn)
            total_loss += config.attention_weight * attention_loss
            metrics["attention_loss"] = attention_loss
        else:
            metrics["attention_loss"] = jnp.array(0.0, dtype=jnp.float32)

        if config.requires_feature_projection():
            if module.projector is not None:
                projected_teacher = module.projector(teacher_hidden)
            else:
                projected_teacher = teacher_hidden
            feature_loss = _feature_projection_loss(student_hidden, projected_teacher)
            total_loss += config.feature_weight * feature_loss
            metrics["feature_loss"] = feature_loss
        else:
            metrics["feature_loss"] = jnp.array(0.0, dtype=jnp.float32)

        preds = jnp.argmax(student_logits, axis=-1)
        accuracy = jnp.mean((preds == batch_y).astype(jnp.float32))
        metrics["accuracy"] = accuracy
        metrics["loss"] = jnp.asarray(total_loss)

        return jnp.asarray(total_loss), metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(container)

    grad_state = nnx.state(grads, nnx.Param)
    param_state = nnx.state(container, nnx.Param)
    updates, opt_state = optimizer.update(grad_state, opt_state, param_state)
    new_params = optax.apply_updates(param_state, updates)
    nnx.update(container, new_params)

    # Ensure scalar metrics are float for logging
    return container, opt_state, {k: float(v) for k, v in metrics.items()}


def load_teacher_checkpoint(checkpoint_dir: Path) -> Tuple[Transformer, CheckpointMetadata]:
    """Load a teacher model from checkpoint for distillation."""

    checkpoint_dir = Path(checkpoint_dir)
    metadata = read_metadata(checkpoint_dir)
    if metadata is None:
        raise ValueError(f"No metadata found in {checkpoint_dir}; cannot load teacher.")

    teacher_config = TransformerConfig(**metadata.config)
    rngs = nnx.Rngs(params=metadata.seed, dropout=metadata.seed)
    teacher = Transformer(teacher_config, rngs)

    optimizer = create_optimizer(
        optimizer_type=metadata.optimizer.get("type", "adamw"),
        learning_rate=metadata.optimizer.get("learning_rate", 1e-3),
        warmup_steps=metadata.optimizer.get("warmup_steps", 0),
        beta1=metadata.optimizer.get("beta1", 0.9),
        beta2=metadata.optimizer.get("beta2", 0.98),
        weight_decay=metadata.optimizer.get("weight_decay", 0.0),
    )

    opt_state = optimizer.init(nnx.state(teacher, nnx.Param))
    rng = jax.random.PRNGKey(metadata.seed)
    restored = restore_checkpoint(checkpoint_dir, teacher, opt_state, rng, metadata)
    if restored is None:
        raise ValueError(f"Unable to restore checkpoint from {checkpoint_dir}.")

    return teacher, metadata


def run_distillation(
    teacher_checkpoint: Path,
    student_config: TransformerConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
    distill_config: DistillationConfig,
    output_dir: Optional[Path] = None,
    seed: int = 0,
) -> DistillationResult:
    """Run knowledge distillation from a teacher checkpoint into a student model."""

    teacher, teacher_metadata = load_teacher_checkpoint(teacher_checkpoint)

    X_train, y_train, X_val, y_val = grokking_data(
        p=data_config.p,
        op=data_config.operation,
        train_fraction=data_config.train_fraction,
        seed=seed,
    )
    n_tokens = teacher_metadata.config["n_tokens"]

    student_rngs = nnx.Rngs(params=seed, dropout=seed)
    student = Transformer(student_config, student_rngs)

    needs_aux = distill_config.requires_intermediates()
    projector: Optional[FeatureProjector] = None
    if distill_config.requires_feature_projection():
        sample_logits, sample_aux = teacher(
            X_train[:1],
            training=False,
            return_intermediates=True,
        )
        num_features = len(sample_aux.get("hidden_states", []))
        _, student_sample_aux = student(
            X_train[:1],
            training=False,
            return_intermediates=True,
        )
        if len(student_sample_aux.get("hidden_states", [])) != num_features:
            raise ValueError(
                "Teacher and student hidden state counts differ; adjust depth scaling."
            )
        projector = FeatureProjector(
            teacher_dim=teacher_metadata.config["dim"],
            student_dim=student_config.dim,
            num_features=num_features,
            rngs=nnx.Rngs(params=seed + 1),
        )

    container = DistillationContainer(student=student, projector=projector)

    optimizer = create_optimizer(
        optimizer_type=training_config.optimizer_type,
        learning_rate=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        beta1=training_config.beta1,
        beta2=training_config.beta2,
        weight_decay=training_config.weight_decay,
    )

    opt_state = optimizer.init(nnx.state(container, nnx.Param))

    history = _create_history_dict()

    output_path = Path(output_dir) if output_dir else None
    history_file = None
    checkpoint_path = None
    metadata = None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        history_file = output_path / "distillation_history.json"
        checkpoint_path = output_path / "checkpoints"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        metadata = CheckpointMetadata(
            config=dataclasses.asdict(student_config),
            optimizer={
                "learning_rate": training_config.learning_rate,
                "beta1": training_config.beta1,
                "beta2": training_config.beta2,
                "weight_decay": training_config.weight_decay,
                "warmup_steps": training_config.warmup_steps,
            },
            seed=seed,
        )

    logger = TunixLogger(project="grokking-distillation", run_name=f"student-dim{student_config.dim}")

    num_train = X_train.shape[0]
    num_batches = math.ceil(num_train / data_config.batch_size)
    total_steps = 0

    for epoch in range(1, training_config.epochs + 1):
        perm = np.random.permutation(num_train)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        epoch_loss = 0.0
        epoch_acc = 0.0
        logit_component = 0.0
        attention_component = 0.0
        feature_component = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * data_config.batch_size
            end = min(start + data_config.batch_size, num_train)
            batch_X = X_train_shuffled[start:end]
            batch_y = y_train_shuffled[start:end]

            container, opt_state, metrics = _distillation_train_step(
                container,
                teacher,
                optimizer,
                opt_state,
                batch_X,
                batch_y,
                n_tokens,
                distill_config,
            )

            batch_size = batch_X.shape[0]
            epoch_loss += metrics.get("loss", 0.0) * batch_size
            epoch_acc += metrics.get("accuracy", 0.0) * batch_size
            logit_component += metrics.get("logit_loss", 0.0) * batch_size
            attention_component += metrics.get("attention_loss", 0.0) * batch_size
            feature_component += metrics.get("feature_loss", 0.0) * batch_size

            total_steps += 1

        train_metrics = {
            "loss": epoch_loss / num_train,
            "accuracy": epoch_acc / num_train,
            "logit_loss": logit_component / num_train,
            "attention_loss": attention_component / num_train,
            "feature_loss": feature_component / num_train,
        }

        val_metrics = _evaluate_student(
            container,
            teacher if needs_aux else None,
            X_val,
            y_val,
            n_tokens,
            distill_config,
        )

        _append_history(history, total_steps, epoch, train_metrics, val_metrics)

        if history_file:
            save_history(history_file, history)

        if checkpoint_path and metadata:
            save_checkpoint(
                checkpoint_path,
                container,
                opt_state,
                jax.random.PRNGKey(seed + epoch),
                total_steps,
                epoch,
                metadata,
            )

        if epoch % training_config.log_every == 0:
            logger.log(
                step=total_steps,
                metrics={
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                },
            )

    final_metrics = {
        "val_loss": history["val_loss"][-1] if history["val_loss"] else 0.0,
        "val_acc": history["val_acc"][-1] if history["val_acc"] else 0.0,
    }

    return DistillationResult(
        history=history,
        final_metrics=final_metrics,
        checkpoint_dir=checkpoint_path,
        student_config=student_config,
    )


def _parse_strategy_list(raw: Sequence[str]) -> Tuple[DistillationStrategy, ...]:
    strategies: List[DistillationStrategy] = []
    for item in raw:
        token = item.strip().lower()
        if not token:
            continue
        strategies.append(DistillationStrategy(token))
    if not strategies:
        strategies.append(DistillationStrategy.LOGIT)
    return tuple(dict.fromkeys(strategies))


def _build_distill_config(args: Any) -> DistillationConfig:
    strategies = _parse_strategy_list(args.strategies)
    return DistillationConfig(
        strategies=strategies,
        temperature=args.temperature,
        alpha=args.alpha,
        attention_weight=args.attention_weight,
        feature_weight=args.feature_weight,
        use_ground_truth=not args.disable_ground_truth,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distill a grokking teacher into a student")
    parser.add_argument("--teacher_checkpoint", type=str, required=True, help="Path to teacher checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store student checkpoints/history")
    parser.add_argument("--student_scale", type=float, default=0.5, help="Width scaling factor for the student")
    parser.add_argument("--depth_scale", type=float, default=1.0, help="Depth scaling factor for the student")
    parser.add_argument(
        "--strategies",
        type=lambda s: [p.strip() for p in s.split(",")],
        default=[DistillationStrategy.LOGIT.value],
        help="Comma separated list of distillation strategies",
    )
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--attention_weight", type=float, default=1.0)
    parser.add_argument("--feature_weight", type=float, default=1.0)
    parser.add_argument("--disable_ground_truth", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--operation", type=str, default="/")
    parser.add_argument("--train_fraction", type=float, default=0.5)

    args = parser.parse_args()

    teacher_metadata = read_metadata(Path(args.teacher_checkpoint))
    if teacher_metadata is None:
        raise SystemExit(f"No metadata.json found in {args.teacher_checkpoint}")

    base_config = TransformerConfig(**teacher_metadata.config)
    student_config = scale_transformer_config(base_config, args.student_scale, args.depth_scale)

    data_config = DataConfig(
        p=args.p,
        operation=args.operation,
        train_fraction=args.train_fraction,
        batch_size=args.batch_size,
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
    )

    distill_config = _build_distill_config(args)

    result = run_distillation(
        teacher_checkpoint=Path(args.teacher_checkpoint),
        student_config=student_config,
        data_config=data_config,
        training_config=training_config,
        distill_config=distill_config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        seed=args.seed,
    )

    print("Distillation complete.")
    print(f"  Student dim: {student_config.dim}, depth: {student_config.depth}")
    print(f"  Validation accuracy: {result.final_metrics['val_acc']:.4f}")
    if result.checkpoint_dir:
        metadata = read_metadata(result.checkpoint_dir)
        latest = result.checkpoint_dir / f"step_{metadata.latest_step:08d}" if metadata else result.checkpoint_dir
        print(f"  Student checkpoints: {latest}")

