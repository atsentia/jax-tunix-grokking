"""Experiment CLI for scaling and distilling grokking models."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from checkpointing import read_metadata
from distillation import (
    DataConfig,
    DistillationConfig,
    DistillationStrategy,
    TrainingConfig,
    run_distillation,
    scale_transformer_config,
)
from models import TransformerConfig
from train_nnx import train
from data import grokking_data


def _ensure_dir(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


def _train_teacher_if_needed(args: argparse.Namespace, base_config: TransformerConfig) -> Path:
    if args.teacher_checkpoint:
        checkpoint_path = Path(args.teacher_checkpoint)
        if not checkpoint_path.exists():
            raise SystemExit(f"Teacher checkpoint directory {checkpoint_path} does not exist")
        metadata = read_metadata(checkpoint_path)
        if metadata is None:
            raise SystemExit(f"Teacher checkpoint at {checkpoint_path} is missing metadata.json")
        return checkpoint_path

    teacher_output = Path(args.output_dir) / f"teacher_scale_{args.teacher_scale:.2f}"
    _ensure_dir(teacher_output)
    checkpoint_dir = teacher_output / "checkpoints"

    teacher_config = scale_transformer_config(
        base_config,
        args.teacher_scale,
        args.teacher_depth_scale,
    )

    print("Training teacher model...")
    train(
        depth=teacher_config.depth,
        dim=teacher_config.dim,
        heads=teacher_config.heads,
        dropout=teacher_config.dropout,
        p=args.p,
        operation=args.operation,
        train_fraction=args.train_fraction,
        batch_size=args.teacher_batch_size,
        epochs=args.teacher_epochs,
        learning_rate=args.teacher_learning_rate,
        weight_decay=args.teacher_weight_decay,
        beta1=args.teacher_beta1,
        beta2=args.teacher_beta2,
        warmup_steps=args.teacher_warmup_steps,
        seed=args.teacher_seed,
        save_dir=str(teacher_output),
        checkpoint_dir=str(checkpoint_dir),
        resume=args.teacher_resume,
    )

    metadata = read_metadata(checkpoint_dir)
    if metadata is None:
        raise SystemExit(f"Teacher training did not produce metadata in {checkpoint_dir}")

    return checkpoint_dir


def _parse_strategies(raw: Sequence[str]) -> List[DistillationStrategy]:
    if not raw:
        return [DistillationStrategy.LOGIT]
    return [DistillationStrategy(token) for token in raw]


def _build_distillation_config(args: argparse.Namespace, strategies: Sequence[DistillationStrategy]) -> DistillationConfig:
    return DistillationConfig(
        strategies=tuple(dict.fromkeys(strategies)),
        temperature=args.distill_temperature,
        alpha=args.distill_alpha,
        attention_weight=args.distill_attention_weight,
        feature_weight=args.distill_feature_weight,
        use_ground_truth=not args.disable_ground_truth,
    )


def _record_manifest(manifest_path: Path, manifest: Dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def _build_training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.distill_epochs,
        learning_rate=args.distill_learning_rate,
        weight_decay=args.distill_weight_decay,
        beta1=args.distill_beta1,
        beta2=args.distill_beta2,
        warmup_steps=args.distill_warmup_steps,
        log_every=args.log_every,
    )


def _build_data_config(args: argparse.Namespace) -> DataConfig:
    return DataConfig(
        p=args.p,
        operation=args.operation,
        train_fraction=args.train_fraction,
        batch_size=args.distill_batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scaling + distillation experiments")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory for experiment artifacts")
    parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Use existing teacher checkpoint")
    parser.add_argument("--teacher_scale", type=float, default=1.0, help="Width scale for training teacher if needed")
    parser.add_argument("--teacher_depth_scale", type=float, default=1.0)
    parser.add_argument("--student_scales", type=float, nargs="+", default=[0.5], help="Student width scale factors relative to teacher")
    parser.add_argument("--student_depth_scale", type=float, default=1.0, help="Student depth scale relative to parent")
    parser.add_argument("--distillation_strategies", type=str, nargs="+", default=[DistillationStrategy.LOGIT.value])
    parser.add_argument("--grokking_threshold", type=float, default=0.99)
    parser.add_argument("--recursive", action="store_true", help="Enable recursive distillation when students grok")
    parser.add_argument("--recursive_factor", type=float, default=0.5, help="Additional shrink factor for recursive rounds")
    parser.add_argument("--max_recursive_rounds", type=int, default=2, help="Maximum recursive rounds per strategy/scale")
    parser.add_argument("--log_every", type=int, default=1)

    # Dataset args
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--operation", type=str, default="/")
    parser.add_argument("--train_fraction", type=float, default=0.5)

    # Base model shape
    parser.add_argument("--base_dim", type=int, default=128)
    parser.add_argument("--base_depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pool", type=str, default="cls")

    # Teacher training hyper-parameters
    parser.add_argument("--teacher_epochs", type=int, default=150)
    parser.add_argument("--teacher_batch_size", type=int, default=512)
    parser.add_argument("--teacher_learning_rate", type=float, default=1e-3)
    parser.add_argument("--teacher_weight_decay", type=float, default=1.0)
    parser.add_argument("--teacher_beta1", type=float, default=0.9)
    parser.add_argument("--teacher_beta2", type=float, default=0.98)
    parser.add_argument("--teacher_warmup_steps", type=int, default=10)
    parser.add_argument("--teacher_seed", type=int, default=42)
    parser.add_argument("--teacher_resume", action="store_true")

    # Student distillation hyper-parameters
    parser.add_argument("--distill_epochs", type=int, default=50)
    parser.add_argument("--distill_batch_size", type=int, default=512)
    parser.add_argument("--distill_learning_rate", type=float, default=5e-4)
    parser.add_argument("--distill_weight_decay", type=float, default=0.0)
    parser.add_argument("--distill_beta1", type=float, default=0.9)
    parser.add_argument("--distill_beta2", type=float, default=0.98)
    parser.add_argument("--distill_warmup_steps", type=int, default=0)
    parser.add_argument("--distill_alpha", type=float, default=0.5)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--distill_attention_weight", type=float, default=1.0)
    parser.add_argument("--distill_feature_weight", type=float, default=1.0)
    parser.add_argument("--disable_ground_truth", action="store_true")

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    output_dir = _ensure_dir(Path(args.output_dir))
    if output_dir is None:
        raise SystemExit("output_dir must be specified")

    X_train, y_train, X_val, y_val = grokking_data(
        p=args.p,
        op=args.operation,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    seq_len = X_train.shape[1]
    n_tokens = args.p + 2

    base_config = TransformerConfig(
        depth=args.base_depth,
        dim=args.base_dim,
        heads=args.heads,
        n_tokens=n_tokens,
        seq_len=seq_len,
        dropout=args.dropout,
        pool=args.pool,
    )

    teacher_checkpoint = _train_teacher_if_needed(args, base_config)
    teacher_metadata = read_metadata(teacher_checkpoint)
    if teacher_metadata is None:
        raise SystemExit(f"Missing metadata at {teacher_checkpoint}")

    teacher_config = TransformerConfig(**teacher_metadata.config)

    strategies = _parse_strategies(args.distillation_strategies)
    distill_config_template = _build_distillation_config(args, strategies)
    training_config = _build_training_config(args)
    data_config = _build_data_config(args)

    manifest = {
        "teacher": {
            "checkpoint": str(teacher_checkpoint),
            "config": teacher_metadata.config,
            "optimizer": teacher_metadata.optimizer,
        },
        "students": [],
    }

    max_rounds = args.max_recursive_rounds if args.recursive else 0

    for base_scale in args.student_scales:
        for strategy in strategies:
            current_teacher_checkpoint = teacher_checkpoint
            current_teacher_config = teacher_config
            overall_scale = base_scale
            for round_idx in range(max_rounds + 1):
                local_scale = base_scale if round_idx == 0 else args.recursive_factor
                effective_scale = overall_scale if round_idx == 0 else overall_scale * args.recursive_factor

                student_config = scale_transformer_config(
                    current_teacher_config,
                    local_scale,
                    args.student_depth_scale,
                )

                student_dir = output_dir / (
                    f"student_strategy-{strategy.value}_scale-{effective_scale:.3f}_round-{round_idx}"
                )
                _ensure_dir(student_dir)

                distill_config = DistillationConfig(
                    strategies=(strategy,),
                    temperature=distill_config_template.temperature,
                    alpha=distill_config_template.alpha,
                    attention_weight=distill_config_template.attention_weight,
                    feature_weight=distill_config_template.feature_weight,
                    use_ground_truth=distill_config_template.use_ground_truth,
                )

                print(
                    f"\nDistilling strategy={strategy.value}, overall_scale={effective_scale:.3f}, round={round_idx}"
                )

                result = run_distillation(
                    teacher_checkpoint=current_teacher_checkpoint,
                    student_config=student_config,
                    data_config=data_config,
                    training_config=training_config,
                    distill_config=distill_config,
                    output_dir=student_dir,
                    seed=args.seed + round_idx,
                )

                val_acc = result.final_metrics.get("val_acc", 0.0)
                success = val_acc >= args.grokking_threshold

                manifest_entry = {
                    "strategy": strategy.value,
                    "round": round_idx,
                    "parent_checkpoint": str(current_teacher_checkpoint),
                    "student_config": dataclasses.asdict(student_config),
                    "overall_scale": effective_scale,
                    "val_acc": val_acc,
                    "checkpoint": str(result.checkpoint_dir) if result.checkpoint_dir else None,
                    "success": success,
                }
                manifest["students"].append(manifest_entry)

                if result.checkpoint_dir is not None:
                    student_metadata = read_metadata(result.checkpoint_dir)
                    if student_metadata is not None:
                        manifest_entry["optimizer"] = student_metadata.optimizer

                if success and args.recursive and result.checkpoint_dir is not None and round_idx < max_rounds:
                    current_teacher_checkpoint = result.checkpoint_dir
                    current_teacher_config = student_config
                    overall_scale *= args.recursive_factor
                else:
                    break

    manifest_path = output_dir / "experiment_manifest.json"
    _record_manifest(manifest_path, manifest)

    print("\nExperiment summary saved to", manifest_path)


if __name__ == "__main__":
    main()

