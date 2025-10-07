"""
Grokking detection utilities.

Detects when a model exhibits the grokking phenomenon:
- Sudden jump in validation accuracy
- Transition from memorization to generalization
"""

from typing import List, Optional, Tuple
import numpy as np


class GrokkingDetector:
    """
    Detector for the grokking phenomenon.

    Grokking is characterized by:
    1. Extended period of high train acc but low val acc (memorization)
    2. Sudden jump in val acc (>20% within a few epochs)
    3. Convergence to high val acc (>95%)
    """

    def __init__(
        self,
        min_jump_threshold: float = 0.20,  # 20% accuracy jump
        window_size: int = 5,  # Look at jumps over 5 epochs
        min_final_acc: float = 0.90,  # Must eventually reach 90% val acc
        min_memorization_epochs: int = 10  # Must show memorization for at least 10 epochs
    ):
        """
        Initialize grokking detector.

        Args:
            min_jump_threshold: Minimum jump in val acc to consider grokking
            window_size: Number of epochs to look for jump
            min_final_acc: Minimum final val acc to confirm grokking
            min_memorization_epochs: Minimum epochs showing memorization phase
        """
        self.min_jump_threshold = min_jump_threshold
        self.window_size = window_size
        self.min_final_acc = min_final_acc
        self.min_memorization_epochs = min_memorization_epochs

        # State
        self.grokking_detected = False
        self.grokking_epoch = None
        self.val_acc_history = []
        self.train_acc_history = []

    def update(self, epoch: int, train_acc: float, val_acc: float) -> bool:
        """
        Update detector with new epoch data.

        Args:
            epoch: Current epoch
            train_acc: Training accuracy
            val_acc: Validation accuracy

        Returns:
            True if grokking just detected, False otherwise
        """
        self.val_acc_history.append(val_acc)
        self.train_acc_history.append(train_acc)

        # Don't detect until we have enough history
        if len(self.val_acc_history) < self.window_size + self.min_memorization_epochs:
            return False

        # Already detected
        if self.grokking_detected:
            return False

        # Check for sudden jump in validation accuracy
        recent_val = np.array(self.val_acc_history[-self.window_size:])
        prev_val = np.array(self.val_acc_history[-(self.window_size * 2):-self.window_size])

        recent_mean = np.mean(recent_val)
        prev_mean = np.mean(prev_val)

        jump = recent_mean - prev_mean

        # Check if there was a significant jump
        if jump >= self.min_jump_threshold:
            # Check if there was a memorization phase (high train acc, low val acc)
            early_train = np.array(self.train_acc_history[:self.min_memorization_epochs])
            early_val = np.array(self.val_acc_history[:self.min_memorization_epochs])

            # Memorization: train acc > 0.5, val acc < 0.5
            had_memorization = np.mean(early_train) > 0.5 and np.mean(early_val) < 0.5

            if had_memorization or len(self.val_acc_history) > 20:
                # Grokking detected!
                self.grokking_detected = True
                self.grokking_epoch = epoch - self.window_size // 2  # Approximate midpoint
                return True

        return False

    def is_grokking(self) -> bool:
        """Check if grokking has been detected."""
        return self.grokking_detected

    def get_grokking_epoch(self) -> Optional[int]:
        """Get the epoch when grokking occurred."""
        return self.grokking_epoch

    def get_status(self) -> str:
        """Get current status as a string."""
        if not self.val_acc_history:
            return "No data"

        current_val = self.val_acc_history[-1]

        if self.grokking_detected:
            return f"Grokked at epoch {self.grokking_epoch}"
        elif len(self.val_acc_history) < self.min_memorization_epochs:
            return "Warming up"
        elif current_val < 0.3:
            return "Memorizing"
        elif current_val < 0.7:
            return "Learning"
        else:
            return "Converging"


def detect_grokking_from_history(
    history: dict,
    min_jump_threshold: float = 0.20,
    window_size: int = 5
) -> Tuple[bool, Optional[int], Optional[float]]:
    """
    Detect grokking from a complete training history.

    Args:
        history: Training history dict with 'epoch', 'train_acc', 'val_acc'
        min_jump_threshold: Minimum jump threshold
        window_size: Window size for jump detection

    Returns:
        Tuple of (grokking_detected, grokking_epoch, grokking_val_acc)
    """
    epochs = history['epoch']
    train_accs = history['train_acc']
    val_accs = history['val_acc']

    if len(epochs) < window_size * 2:
        return False, None, None

    # Look for the largest jump in validation accuracy
    max_jump = 0
    max_jump_epoch = None
    max_jump_val_acc = None

    for i in range(window_size, len(val_accs) - window_size):
        prev_val = np.mean(val_accs[i - window_size:i])
        next_val = np.mean(val_accs[i:i + window_size])

        jump = next_val - prev_val

        if jump > max_jump:
            max_jump = jump
            max_jump_epoch = epochs[i]
            max_jump_val_acc = val_accs[i]

    # Check if jump was significant enough
    if max_jump >= min_jump_threshold:
        return True, max_jump_epoch, max_jump_val_acc
    else:
        return False, None, None


def analyze_training_phases(history: dict) -> dict:
    """
    Analyze training phases from history.

    Identifies:
    - Memorization phase: high train acc, low val acc
    - Transition phase: rapid improvement in val acc
    - Generalization phase: high train and val acc

    Args:
        history: Training history dict

    Returns:
        Dictionary with phase information
    """
    train_accs = np.array(history['train_acc'])
    val_accs = np.array(history['val_acc'])
    epochs = np.array(history['epoch'])

    # Find memorization phase (train > 0.6, val < 0.4)
    memorization_mask = (train_accs > 0.6) & (val_accs < 0.4)
    memorization_epochs = epochs[memorization_mask]

    # Find generalization phase (both > 0.9)
    generalization_mask = (train_accs > 0.9) & (val_accs > 0.9)
    generalization_epochs = epochs[generalization_mask]

    # Find transition phase (between memorization and generalization)
    transition_mask = ~memorization_mask & ~generalization_mask
    transition_epochs = epochs[transition_mask]

    # Calculate phase durations
    phases = {
        'memorization': {
            'epochs': memorization_epochs.tolist() if len(memorization_epochs) > 0 else [],
            'duration': len(memorization_epochs),
            'avg_train_acc': np.mean(train_accs[memorization_mask]) if np.any(memorization_mask) else None,
            'avg_val_acc': np.mean(val_accs[memorization_mask]) if np.any(memorization_mask) else None
        },
        'transition': {
            'epochs': transition_epochs.tolist() if len(transition_epochs) > 0 else [],
            'duration': len(transition_epochs)
        },
        'generalization': {
            'epochs': generalization_epochs.tolist() if len(generalization_epochs) > 0 else [],
            'duration': len(generalization_epochs),
            'avg_train_acc': np.mean(train_accs[generalization_mask]) if np.any(generalization_mask) else None,
            'avg_val_acc': np.mean(val_accs[generalization_mask]) if np.any(generalization_mask) else None
        }
    }

    # Detect grokking
    grokked, grok_epoch, grok_val_acc = detect_grokking_from_history(history)

    phases['grokking'] = {
        'detected': grokked,
        'epoch': grok_epoch,
        'val_acc': grok_val_acc
    }

    return phases


def print_phase_analysis(phases: dict):
    """
    Print a formatted phase analysis.

    Args:
        phases: Phase analysis dictionary from analyze_training_phases
    """
    print("=" * 80)
    print("Training Phase Analysis")
    print("=" * 80)

    # Memorization phase
    mem = phases['memorization']
    if mem['duration'] > 0:
        print(f"\nMemorization Phase: {mem['duration']} epochs")
        print(f"  Epochs: {mem['epochs'][0]} - {mem['epochs'][-1]}")
        print(f"  Avg Train Acc: {mem['avg_train_acc']*100:.1f}%")
        print(f"  Avg Val Acc: {mem['avg_val_acc']*100:.1f}%")
    else:
        print("\nMemorization Phase: Not detected")

    # Transition phase
    trans = phases['transition']
    if trans['duration'] > 0:
        print(f"\nTransition Phase: {trans['duration']} epochs")
        if len(trans['epochs']) > 0:
            print(f"  Epochs: {trans['epochs'][0]} - {trans['epochs'][-1]}")
    else:
        print("\nTransition Phase: Not detected")

    # Generalization phase
    gen = phases['generalization']
    if gen['duration'] > 0:
        print(f"\nGeneralization Phase: {gen['duration']} epochs")
        print(f"  Epochs: {gen['epochs'][0]} - {gen['epochs'][-1]}")
        print(f"  Avg Train Acc: {gen['avg_train_acc']*100:.1f}%")
        print(f"  Avg Val Acc: {gen['avg_val_acc']*100:.1f}%")
    else:
        print("\nGeneralization Phase: Not detected")

    # Grokking
    grok = phases['grokking']
    print(f"\nGrokking: {'Yes' if grok['detected'] else 'No'}")
    if grok['detected']:
        print(f"  Epoch: {grok['epoch']}")
        print(f"  Val Acc at Grokking: {grok['val_acc']*100:.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    # Test grokking detector
    print("Testing grokking detector...")

    # Simulate a grokking training run
    detector = GrokkingDetector(min_jump_threshold=0.20, window_size=5)

    # Memorization phase (epochs 0-30)
    for epoch in range(30):
        train_acc = 0.7 + 0.01 * epoch
        val_acc = 0.2 + 0.003 * epoch
        just_grokked = detector.update(epoch, train_acc, val_acc)
        if just_grokked:
            print(f"✓ Grokking detected at epoch {epoch}!")

    # Transition phase (epochs 30-40)
    for epoch in range(30, 40):
        train_acc = 0.9
        val_acc = 0.3 + (epoch - 30) * 0.07  # Rapid jump
        just_grokked = detector.update(epoch, train_acc, val_acc)
        if just_grokked:
            print(f"✓ Grokking detected at epoch {epoch}!")

    # Generalization phase (epochs 40-60)
    for epoch in range(40, 60):
        train_acc = 0.98
        val_acc = 0.95
        just_grokked = detector.update(epoch, train_acc, val_acc)

    print(f"\nFinal status: {detector.get_status()}")
    print(f"Grokking epoch: {detector.get_grokking_epoch()}")

    # Test history analysis
    print("\n" + "=" * 80)
    print("Testing history analysis...")

    # Create synthetic history
    history = {
        'epoch': list(range(60)),
        'train_acc': [0.7 + 0.01 * i if i < 30 else 0.9 if i < 40 else 0.98 for i in range(60)],
        'val_acc': [0.2 + 0.003 * i if i < 30 else 0.3 + (i - 30) * 0.07 if i < 40 else 0.95 for i in range(60)]
    }

    phases = analyze_training_phases(history)
    print_phase_analysis(phases)

    print("\n✓ Grokking detector test completed!")
