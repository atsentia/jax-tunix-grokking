# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Muon Optimizer Support**: Added support for the Muon optimizer as an alternative to AdamW
  - Created `src/optimizers.py` with factory functions for both AdamW and Muon optimizers
  - Added `--optimizer {adamw|muon}` CLI argument to `train_nnx.py`
  - Muon uses Newton-Schulz orthogonalization for more efficient training, especially with larger batch sizes
  - Both optimizers support linear warmup and weight decay

### Changed
- **Model Architecture Updates**: Modified `src/models.py` to use Python lists instead of `nnx.List()` for storing transformer layers
  - Fixed compatibility issue with current Flax NNX API
  - Changed from separate `attention_layers` and `ffn_layers` lists to single `layers` list of tuples

### Fixed
- Fixed `AttributeError` when instantiating Transformer model due to `nnx.List()` not being available in Flax NNX
- Fixed Muon optimizer initialization by removing unsupported `momentum_dtype` parameter

## [0.2.0] - 2025-10-07

### Added
- **Distillation Support**: Complete knowledge distillation implementation using Tunix
  - Orbax-based checkpointing with metadata (`src/checkpointing.py`)
  - Multiple distillation strategies: logit, attention transfer, feature projection (`src/distillation.py`)
  - Experiment orchestration script (`src/run_experiments.py`)
  - Product requirements document (`docs/prd_grokking_distillation.md`)
- **Checkpoint Resume**: Training can now resume from Orbax checkpoints
  - Added `--checkpoint_dir` and `--resume` CLI arguments
  - Automatic state restoration for model, optimizer, and RNG
- **Enhanced Model API**: Added optional intermediate outputs
  - `return_attention` parameter for attention weights
  - `return_intermediates` parameter for hidden states and attention maps

### Changed
- Updated `pyproject.toml` dependencies:
  - Added `orbax-checkpoint>=0.5.0`
  - Added `tunix` from GitHub (not on PyPI)
- Enhanced `train_nnx.py` with checkpointing and history persistence
  - Training history saved incrementally during training
  - Metadata tracking for reproducibility

### Fixed
- Improved checkpoint serialization handling for NNX models
- Better error handling for checkpoint loading

## [0.1.0] - 2025-10-06

### Added
- Initial implementation of grokking phenomenon in JAX/Flax NNX
- Transformer model with modern components:
  - RMSNorm layer normalization
  - Rotary Position Embeddings (RoPE)
  - Causal self-attention
  - SiLU-gated feedforward networks
- Modular arithmetic dataset generation (addition, subtraction, multiplication, division)
- Training script with AdamW optimizer and linear warmup
- Configuration files for different experiment settings
- Comprehensive test suite
- Visualization utilities for training curves
- Example training results demonstrating grokking phenomenon

### Documentation
- Detailed README with installation instructions
- Architecture description and hyperparameter guidelines
- Citation information and attributions
- TPU training instructions for Google Colab

## References

- **Grokking Paper**: [Power et al., 2022](https://arxiv.org/abs/2201.02177)
- **Muon Optimizer**: [Keller Jordan's blog](https://kellerjordan.github.io/posts/muon/)
- **Original MLX Implementation**: [stockeh/mlx-grokking](https://github.com/stockeh/mlx-grokking)
