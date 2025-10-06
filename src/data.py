# data.py - Grokking Dataset for Tunix
# Ported from jax_grokking/data.py with 100% parity in data generation logic

import numpy as np
import jax.numpy as jnp
from typing import Tuple, Iterator, Dict
import dataclasses


def grokking_data(p: int, op: str = '/', train_fraction: float = 0.5, seed: int = 42):
    """
    Generate training and test data for modular arithmetic grokking.

    CRITICAL: This function preserves 100% parity with jax_grokking/data.py
    to ensure exact reproducibility of the grokking phenomenon.

    Args:
        p: prime modulus for arithmetic (e.g., 97)
        op: operation in {'*','/','+','-'} (defaults to '/')
        train_fraction: fraction of data used for training (remainder is validation)
        seed: random seed for reproducibility

    Returns:
        X_train, y_train, X_test, y_test as JAX arrays (int32).

    Raises:
        ValueError: if op is not in supported operations
        AssertionError: if data validation fails
    """
    # Supported operations (results mod p)
    operations = {
        '*': lambda a, b: (a * b) % p,
        '/': lambda a, b: (a * pow(int(b), p - 2, p)) % p,  # Fermat's little theorem
        '+': lambda a, b: (a + b) % p,
        '-': lambda a, b: (a - b) % p
    }

    assert op in operations, f"Unsupported operation '{op}'. Choose from {list(operations.keys())}"

    # Generate all pairs (a, b), excluding b=0 for division
    b_start = 1 if op == '/' else 0
    pairs = [(a, b) for a in range(p) for b in range(b_start, p)]
    results = [operations[op](a, b) for (a, b) in pairs]

    pairs = np.array(pairs, dtype=int)
    results = np.array(results, dtype=int)

    # Encode input sequences [a, op_token, b, equals_token]
    op_token = p      # ID for the operation
    eq_token = p + 1  # ID for '='
    seqs = np.stack([
        pairs[:, 0],                     # a
        np.full(len(pairs), op_token),   # op
        pairs[:, 1],                     # b
        np.full(len(pairs), eq_token)    # '='
    ], axis=1)

    # Assertions for data validity
    assert seqs.shape[1] == 4, f"Expected sequence length 4, got {seqs.shape[1]}"
    assert seqs.shape[0] == len(results), "Mismatch between sequences and results"
    assert np.all(seqs[:, 1] == op_token), "Op token should be consistent"
    assert np.all(seqs[:, 3] == eq_token), "Equals token should be consistent"

    # Shuffle and split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(seqs))
    n_train = int(train_fraction * len(seqs))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, y_train = seqs[train_idx], results[train_idx]
    X_test,  y_test  = seqs[test_idx],  results[test_idx]

    # Convert to JAX arrays (int32) - matches baseline exactly
    X_train = jnp.array(X_train, dtype=jnp.int32)
    y_train = jnp.array(y_train, dtype=jnp.int32)
    X_test  = jnp.array(X_test,  dtype=jnp.int32)
    y_test  = jnp.array(y_test,  dtype=jnp.int32)

    # Final assertions
    assert X_train.shape[0] == n_train, f"Training set size mismatch: {X_train.shape[0]} vs {n_train}"
    assert X_test.shape[0] == len(seqs) - n_train, "Test set size mismatch"
    assert X_train.dtype == jnp.int32, f"X_train dtype should be int32, got {X_train.dtype}"
    assert y_train.dtype == jnp.int32, f"y_train dtype should be int32, got {y_train.dtype}"

    return X_train, y_train, X_test, y_test


@dataclasses.dataclass
class GrokkingDatasetConfig:
    """Configuration for grokking dataset."""
    p: int = 97                    # Prime modulus
    operation: str = '/'           # Arithmetic operation
    train_fraction: float = 0.5    # Fraction of data for training
    seed: int = 42                 # Random seed
    batch_size: int = 512          # Batch size for training


class GrokkingDatasetIterator:
    """
    Iterator wrapper for grokking data compatible with Tunix training loops.

    This iterator provides batches in a format expected by Tunix trainers,
    with proper teacher-forcing for the causal language model setup.
    """

    def __init__(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize dataset iterator.

        Args:
            X: Input sequences [N, seq_len]
            y: Target values [N]
            batch_size: Batch size
            shuffle: Whether to shuffle data each epoch
            seed: Random seed for shuffling
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.n_samples = X.shape[0]
        self.n_batches = int(np.ceil(self.n_samples / batch_size))
        self.rng = np.random.RandomState(seed)
        self._reset()

    def _reset(self):
        """Reset iterator to beginning."""
        if self.shuffle:
            perm = self.rng.permutation(self.n_samples)
            self.X_shuffled = self.X[perm]
            self.y_shuffled = self.y[perm]
        else:
            self.X_shuffled = self.X
            self.y_shuffled = self.y
        self.current_idx = 0

    def __iter__(self):
        self._reset()
        return self

    def __next__(self) -> Dict[str, jnp.ndarray]:
        """
        Get next batch.

        Returns:
            Dictionary with 'input_ids' and 'labels' keys.
        """
        if self.current_idx >= self.n_samples:
            raise StopIteration

        start_idx = self.current_idx
        end_idx = min(start_idx + self.batch_size, self.n_samples)

        batch_X = self.X_shuffled[start_idx:end_idx]
        batch_y = self.y_shuffled[start_idx:end_idx]

        self.current_idx = end_idx

        # Return in format expected by Tunix
        # input_ids: the input sequence (for our case, the arithmetic expression)
        # labels: the target result (what we're predicting)
        return {
            'input_ids': batch_X,
            'labels': batch_y,
        }

    def __len__(self) -> int:
        """Return number of batches."""
        return self.n_batches


def create_grokking_datasets(
    config: GrokkingDatasetConfig
) -> Tuple[GrokkingDatasetIterator, GrokkingDatasetIterator]:
    """
    Create train and validation dataset iterators for grokking.

    Args:
        config: Dataset configuration

    Returns:
        Tuple of (train_iterator, val_iterator)
    """
    # Generate data using the core grokking_data function (100% parity)
    X_train, y_train, X_val, y_val = grokking_data(
        p=config.p,
        op=config.operation,
        train_fraction=config.train_fraction,
        seed=config.seed
    )

    # Create iterators
    train_iter = GrokkingDatasetIterator(
        X_train, y_train,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed
    )

    val_iter = GrokkingDatasetIterator(
        X_val, y_val,
        batch_size=config.batch_size,
        shuffle=False,  # No shuffling for validation
        seed=config.seed
    )

    return train_iter, val_iter


if __name__ == "__main__":
    # Quick test/verification
    print("Testing Grokking Data Generation...")

    # Test basic data generation
    X_train, y_train, X_val, y_val = grokking_data(p=7, op='+', train_fraction=0.7, seed=42)

    print(f"Train set size: {X_train.shape[0]}")
    print(f"Val set size: {X_val.shape[0]}")
    print(f"Input sequence length: {X_train.shape[1]}")
    print(f"Sample input: {X_train[0]}")
    print(f"Sample label: {y_train[0]}")
    print(f"Vocabulary size: {jnp.max(X_train) + 1}  (0 to p+1, p={7})")

    # Test iterator
    config = GrokkingDatasetConfig(p=7, operation='+', batch_size=4, seed=42)
    train_iter, val_iter = create_grokking_datasets(config)

    print(f"\nDataset iterator test:")
    print(f"Number of train batches: {len(train_iter)}")
    print(f"Number of val batches: {len(val_iter)}")

    # Get first batch
    first_batch = next(iter(train_iter))
    print(f"First batch keys: {first_batch.keys()}")
    print(f"Batch input_ids shape: {first_batch['input_ids'].shape}")
    print(f"Batch labels shape: {first_batch['labels'].shape}")

    print("\n✓ Data generation test passed!")
