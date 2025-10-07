# Product Requirements Document: Grokking Checkpointing and Distillation Pipeline

## 1. Overview
- **Product**: Grokking training + distillation toolkit built on JAX/Flax NNX and Tunix
- **Document Owner**: ChatGPT (on behalf of repository maintainers)
- **Stakeholders**: Research engineers experimenting with grokking, ML practitioners exploring distillation, open-source contributors
- **Status**: Draft – implementation tracked in this change

## 2. Background & Context
Existing grokking implementation focuses on baseline training loops. It lacks durable checkpointing, Tunix-based knowledge distillation, and an experiment harness for comparing model scaling factors. Researchers must rerun expensive training after each interruption and cannot easily test how distilled students behave when architectures shrink or grow. Tunix provides reusable distillation strategies (logit, attention transfer, feature projection) that can accelerate experimentation, but the repository does not integrate them yet.

## 3. Goals & Non-Goals
### Goals
1. **Checkpointing** – Persist model + optimizer state using Orbax so grokked models can be resumed or consumed by downstream jobs (e.g., distillation).
2. **Distillation Pipeline** – Provide Tunix-backed scripts that load checkpoints, run multiple distillation strategies, and produce evaluation metrics/history.
3. **Experiment Factors CLI** – Offer a command-line workflow to explore scaling factors (shrink/grow), recursive distillation, and strategy sweeps.

### Non-Goals
- Replacing existing baseline training loops with Tunix Trainer (future work).
- Implementing TPU/accelerator-specific launch scripts.
- Providing pre-trained checkpoints.
- Guaranteeing convergence for every configuration (the framework enables experimentation but does not certify outcomes).

## 4. User Stories
1. *As a researcher*, I can resume grokking training after a crash without losing progress.
2. *As a practitioner*, I can load a grokked teacher checkpoint and distill it into smaller students using Tunix strategies.
3. *As an experimenter*, I can launch CLI experiments that automatically scale architectures and run sequential distillation rounds, stopping when the student stops grokking.

## 5. Success Metrics
- Checkpoint files saved during training and loadable to reproduce final validation accuracy within tolerance.
- Distillation script outputs metrics (loss/accuracy curves) and checkpoints for student models.
- Experiment CLI accepts scale/strategy parameters and emits a run manifest summarizing each trial and whether grokking threshold was met.

## 6. Detailed Requirements
### 6.1 Checkpointing
- Use `orbax.checkpoint` for serialization of model state, optimizer state, RNG key, and progress metadata (epoch/step).
- Persist metadata JSON containing model config, optimizer hyperparameters, seed, and latest step.
- Support resume flag in training CLI to continue from latest checkpoint if available.
- Keep history JSON synced for external plotting.

### 6.2 Distillation
- Load teacher checkpoints via Orbax utility.
- Support **Logit**, **Attention Transfer**, and **Feature Projection** strategies (configurable weights/temperature).
- Implement student training loop leveraging Tunix utilities (dataset iterators, logging) while remaining compatible with NNX models.
- Produce checkpoints/history for students similar to training loop.

### 6.3 Experiment Factors
- CLI to configure:
  - Shrink factor (e.g., 0.5× params) and evaluate student grokking.
  - Grow factor (e.g., 2×–5×) for teacher pretraining before distillation.
  - Strategy sweep across all supported distillation strategies.
  - Recursive distillation: continue shrinking while validation accuracy ≥ threshold.
- Manifest output summarizing each experiment (teacher scale, student scale, strategy, grokking outcome, checkpoint paths).

## 7. UX & CLI Design
- Extend `train_nnx.py` flags: `--checkpoint_dir`, `--resume`.
- New script `distillation.py` with CLI flags for teacher checkpoint, student scaling, strategies, and hyperparameters.
- New orchestrator `run_experiments.py` that composes training + distillation based on user-provided factors.
- Logs include rich progress summaries (epoch, accuracy, distillation losses).

## 8. Technical Approach
1. **Checkpoint Module** – Wrap Orbax `PyTreeCheckpointer` to save/restore training state. Maintain metadata/history JSON sidecar files.
2. **Model Enhancements** – Allow the transformer/attention blocks to emit intermediate activations for distillation objectives.
3. **Distillation Implementation** – Build `DistillationRunner` using Tunix loops for dataset management and incorporate strategy-specific loss components.
4. **Experiment Harness** – Compose training + distillation functions, evaluate grokking threshold, and handle recursive shrinkage.

## 9. Risks & Mitigations
- *Orbax version mismatch*: Pin dependency and add targeted serialization helpers.
- *Distillation stability*: Expose hyperparameters/weights via CLI so users can tune.
- *Recursive experiments explosion*: Provide safeguards (max rounds) and manifest logging.

## 10. Rollout Plan
- Land checkpoint + distillation modules with unit coverage for serialization and intermediate outputs.
- Document new workflow in README (future follow-up) and provide sample commands in docstrings.
- Encourage community feedback on Tunix integration and strategy defaults.

## 11. Open Questions
- Should we adopt Tunix trainers for baseline training? (Deferred.)
- Do we need evaluation metrics beyond accuracy (e.g., weight norms) surfaced in experiment manifest? (Potential enhancement.)

