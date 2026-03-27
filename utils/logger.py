"""
Experiment Logger
=================
Unified logging to TensorBoard and console with timestamp support.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class ExperimentLogger:
    """Combined TensorBoard + console logger for experiment tracking.

    Args:
        log_dir: Directory for TensorBoard logs.
        experiment_name: Name prefix for the experiment.
        use_tensorboard: Enable TensorBoard logging.
        console_level: Console logging level.
    """

    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: str = "experiment",
        use_tensorboard: bool = True,
        console_level: int = logging.INFO,
    ):
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{timestamp}"
        self.log_path = os.path.join(log_dir, self.experiment_id)
        os.makedirs(self.log_path, exist_ok=True)

        # TensorBoard writer
        self.writer = None
        if use_tensorboard and HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=self.log_path)

        # Console logger
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(console_level)

        if not self.logger.handlers:
            # Console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(console_level)
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            # File handler
            fh = logging.FileHandler(
                os.path.join(self.log_path, "training.log")
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)

    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
    ) -> None:
        """Log a scalar value to TensorBoard.

        Args:
            tag: Metric name (e.g., 'train/loss').
            value: Scalar value.
            step: Global step or epoch number.
        """
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict,
        step: int,
    ) -> None:
        """Log multiple scalars under a main tag.

        Args:
            main_tag: Group name.
            tag_scalar_dict: Dict of {name: value}.
            step: Global step.
        """
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(
        self,
        tag: str,
        values,
        step: int,
    ) -> None:
        """Log histogram to TensorBoard.

        Args:
            tag: Histogram name.
            values: Tensor or numpy array of values.
            step: Global step.
        """
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_text(
        self,
        tag: str,
        text: str,
        step: int,
    ) -> None:
        """Log text to TensorBoard.

        Args:
            tag: Text tag.
            text: Text content.
            step: Global step.
        """
        if self.writer:
            self.writer.add_text(tag, text, step)

    def close(self) -> None:
        """Flush and close all writers."""
        if self.writer:
            self.writer.flush()
            self.writer.close()
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
