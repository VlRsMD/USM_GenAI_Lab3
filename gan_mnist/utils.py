"""Small shared helpers: seeding, logging, saving results."""

import json
import logging
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from gan_mnist.trainer import TrainingHistory


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")


def save_history(history: TrainingHistory, path: Path) -> None:
    """Write per-epoch losses to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"d_loss": history.d_loss, "g_loss": history.g_loss}, indent=2))
