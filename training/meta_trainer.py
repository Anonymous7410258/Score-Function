"""
Meta-Trainer Module
===================
Implements the meta-learning training loop for Meta-TTA-TSM.

The meta-training procedure optimizes the hypernetwork H_ϕ so that,
after K inner-loop gradient steps on task-specific data, the
resulting score functions generalize well across diverse topologies.

Algorithm:
    1. Sample batch of tasks {T_i} from task distribution.
    2. For each task T_i:
        a. Extract topological features f_T = Φ(P_T).
        b. Generate initial parameters θ_T^(0) = H_ϕ(f_T).
        c. Perform K inner-loop gradient steps on ISM loss.
        d. Compute outer loss L_ISM(θ_T^(K)).
    3. Update hypernetwork ϕ via meta-gradient.

Reference: Section 4.0.4-4.0.5, Equations 15-17 of the paper.
"""

import os
import copy
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List

from models.score_network import ScoreNetwork
from models.hypernetwork import HyperNetwork
from models.topology import TopologyExtractor
from models.losses import ISMLoss
from data.missingness import MissingnessTaskSampler
from utils.logger import ExperimentLogger
from utils.reproducibility import set_seed


class MetaTrainer:
    """Meta-learning trainer for topology-conditioned score matching.

    Implements the outer-loop / inner-loop meta-learning procedure
    described in the paper. The hypernetwork learns to map topological
    features to score network parameters that can be rapidly adapted
    to novel missingness-induced topologies.

    Args:
        score_network: Base score network architecture (template).
        hypernetwork: Hypernetwork H_ϕ to be meta-trained.
        topology_extractor: Topological feature extractor.
        config: Training configuration dictionary.
        device: Torch device.
        logger: Experiment logger for TensorBoard + console.
    """

    def __init__(
        self,
        score_network: ScoreNetwork,
        hypernetwork: HyperNetwork,
        topology_extractor: TopologyExtractor,
        config: dict,
        device: torch.device,
        logger: Optional[ExperimentLogger] = None,
    ):
        self.score_network = score_network.to(device)
        self.hypernetwork = hypernetwork.to(device)
        self.topology_extractor = topology_extractor
        self.config = config
        self.device = device
        self.logger = logger

        # Extract training config
        meta_cfg = config.get("meta_learning", {})
        inner_cfg = config.get("inner_loop", {})
        train_cfg = config.get("training", {})

        self.meta_lr = meta_cfg.get("meta_lr", 1e-3)
        self.inner_lr = inner_cfg.get("inner_lr", 1e-2)
        self.inner_steps = inner_cfg.get("inner_steps", 5)
        self.first_order = inner_cfg.get("first_order", False)
        self.num_meta_epochs = meta_cfg.get("num_meta_epochs", 200)
        self.tasks_per_batch = meta_cfg.get("tasks_per_batch", 8)
        self.gradient_clip = train_cfg.get("gradient_clip_norm", 1.0)
        self.eval_every = train_cfg.get("eval_every", 5)

        # Loss function
        self.ism_loss = ISMLoss(use_hutchinson=True)

        # Meta-optimizer (for hypernetwork parameters)
        optimizer_name = meta_cfg.get("meta_optimizer", "adam")
        weight_decay = meta_cfg.get("meta_weight_decay", 1e-5)

        if optimizer_name == "adam":
            self.meta_optimizer = torch.optim.Adam(
                self.hypernetwork.parameters(),
                lr=self.meta_lr,
                weight_decay=weight_decay,
            )
        else:
            self.meta_optimizer = torch.optim.SGD(
                self.hypernetwork.parameters(),
                lr=self.meta_lr,
                weight_decay=weight_decay,
                momentum=0.9,
            )

        # Learning rate scheduler
        scheduler_type = meta_cfg.get("meta_lr_scheduler", "cosine")
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.meta_optimizer,
                T_max=self.num_meta_epochs,
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.meta_optimizer,
                step_size=50,
                gamma=0.5,
            )
        else:
            self.scheduler = None

        # Training state
        self.global_step = 0
        self.best_meta_loss = float("inf")

    def _inner_loop(
        self,
        score_net: ScoreNetwork,
        task_data: torch.Tensor,
        task_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Perform K inner-loop gradient steps on a single task.

        Adapts the score network parameters using implicit score matching
        loss on the task-specific data (Equation 16).

        Args:
            score_net: Score network with initialized parameters θ^(0).
            task_data: Observed data (samples_per_task, d).
            task_masks: Binary masks (samples_per_task, d).

        Returns:
            The ISM loss after K inner-loop steps (for meta-gradient).
        """
        # Create parameter list for inner-loop optimization
        inner_params = list(score_net.parameters())

        for k in range(self.inner_steps):
            # Compute ISM loss on task data
            loss = self.ism_loss(score_net, task_data, task_masks)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                inner_params,
                create_graph=not self.first_order,
                retain_graph=True,
                allow_unused=True,
            )

            # SGD update (inner loop)
            with torch.no_grad() if self.first_order else torch.enable_grad():
                for param, grad in zip(inner_params, grads):
                    if grad is not None:
                        param.data = param - self.inner_lr * grad

        # Compute final loss after K steps for meta-gradient
        final_loss = self.ism_loss(score_net, task_data, task_masks)
        return final_loss

    def _meta_step(
        self,
        task_sampler: MissingnessTaskSampler,
    ) -> Dict[str, float]:
        """Perform one meta-learning step over a batch of tasks.

        Reference: Equation 17 — meta-objective optimization.

        Args:
            task_sampler: Sampler providing diverse missingness tasks.

        Returns:
            Dictionary with meta-training metrics.
        """
        self.meta_optimizer.zero_grad()

        # Sample batch of tasks
        task_batch = task_sampler.sample_batch(self.tasks_per_batch)

        total_outer_loss = 0.0
        task_losses = []

        for task in task_batch:
            # Convert to tensors
            task_data = torch.tensor(
                task["observed_data"], dtype=torch.float32, device=self.device
            )
            task_masks = torch.tensor(
                task["masks"], dtype=torch.float32, device=self.device
            )

            # Step 1: Extract topological features f_T = Φ(P_T)
            topo_result = self.topology_extractor.extract(
                task["observed_data"]
            )
            topo_features = torch.tensor(
                topo_result["features"],
                dtype=torch.float32,
                device=self.device,
            )

            # Step 2: Generate initial parameters θ_T^(0) = H_ϕ(f_T)
            generated_params = self.hypernetwork(topo_features)

            # Step 3: Create task-specific score network
            task_score_net = copy.deepcopy(self.score_network)
            self.hypernetwork.apply_params_to_network(
                task_score_net, generated_params
            )

            # Step 4: Inner-loop adaptation (K gradient steps)
            outer_loss = self._inner_loop(
                task_score_net, task_data, task_masks
            )

            total_outer_loss = total_outer_loss + outer_loss
            task_losses.append(outer_loss.item())

        # Average over tasks
        meta_loss = total_outer_loss / len(task_batch)

        # Meta-gradient step
        meta_loss.backward()

        # Gradient clipping
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(
                self.hypernetwork.parameters(),
                self.gradient_clip,
            )

        self.meta_optimizer.step()

        return {
            "meta_loss": meta_loss.item(),
            "task_losses_mean": np.mean(task_losses),
            "task_losses_std": np.std(task_losses),
        }

    def train(
        self,
        task_sampler: MissingnessTaskSampler,
        val_task_sampler: Optional[MissingnessTaskSampler] = None,
        checkpoint_dir: str = "checkpoints",
    ) -> Dict[str, list]:
        """Run meta-training loop.

        Args:
            task_sampler: Training task sampler.
            val_task_sampler: Optional validation task sampler.
            checkpoint_dir: Directory to save checkpoints.

        Returns:
            Dictionary with training history.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        history = {
            "meta_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        if self.logger:
            self.logger.info(
                f"Starting meta-training for {self.num_meta_epochs} epochs"
            )
            self.logger.info(
                f"  Inner steps: {self.inner_steps}, "
                f"Tasks/batch: {self.tasks_per_batch}"
            )

        for epoch in range(1, self.num_meta_epochs + 1):
            self.hypernetwork.train()
            epoch_start = time.time()

            # Meta-training step
            metrics = self._meta_step(task_sampler)
            self.global_step += 1

            # Record history
            history["meta_loss"].append(metrics["meta_loss"])
            current_lr = self.meta_optimizer.param_groups[0]["lr"]
            history["learning_rate"].append(current_lr)

            # Logging
            if self.logger:
                self.logger.log_scalar(
                    "train/meta_loss", metrics["meta_loss"], epoch
                )
                self.logger.log_scalar("train/lr", current_lr, epoch)

                if epoch % self.eval_every == 0 or epoch == 1:
                    elapsed = time.time() - epoch_start
                    self.logger.info(
                        f"Epoch {epoch}/{self.num_meta_epochs} | "
                        f"Meta Loss: {metrics['meta_loss']:.6f} | "
                        f"Task Loss: {metrics['task_losses_mean']:.6f} "
                        f"± {metrics['task_losses_std']:.6f} | "
                        f"LR: {current_lr:.6f} | "
                        f"Time: {elapsed:.1f}s"
                    )

            # Validation
            if val_task_sampler is not None and epoch % self.eval_every == 0:
                val_loss = self._validate(val_task_sampler)
                history["val_loss"].append(val_loss)

                if self.logger:
                    self.logger.log_scalar("val/meta_loss", val_loss, epoch)
                    self.logger.info(f"  Val Loss: {val_loss:.6f}")

                # Save best model
                if val_loss < self.best_meta_loss:
                    self.best_meta_loss = val_loss
                    self._save_checkpoint(
                        os.path.join(checkpoint_dir, "best_model.pt"),
                        epoch,
                        val_loss,
                    )

            # Periodic checkpoint
            save_every = self.config.get("output", {}).get("save_every", 10)
            if epoch % save_every == 0:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt"),
                    epoch,
                    metrics["meta_loss"],
                )

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final model
        self._save_checkpoint(
            os.path.join(checkpoint_dir, "final_model.pt"),
            self.num_meta_epochs,
            metrics["meta_loss"],
        )

        if self.logger:
            self.logger.info("Meta-training complete.")

        return history

    @torch.no_grad()
    def _validate(
        self,
        val_task_sampler: MissingnessTaskSampler,
        n_val_tasks: int = 20,
    ) -> float:
        """Evaluate meta-model on validation tasks.

        Args:
            val_task_sampler: Validation task sampler.
            n_val_tasks: Number of validation tasks.

        Returns:
            Mean validation ISM loss.
        """
        self.hypernetwork.eval()
        val_losses = []

        for _ in range(n_val_tasks):
            task = val_task_sampler.sample_task()

            task_data = torch.tensor(
                task["observed_data"], dtype=torch.float32, device=self.device
            )
            task_masks = torch.tensor(
                task["masks"], dtype=torch.float32, device=self.device
            )

            # Extract topology and generate parameters
            topo_result = self.topology_extractor.extract(
                task["observed_data"]
            )
            topo_features = torch.tensor(
                topo_result["features"],
                dtype=torch.float32,
                device=self.device,
            )

            generated_params = self.hypernetwork(topo_features)
            task_score_net = copy.deepcopy(self.score_network)
            self.hypernetwork.apply_params_to_network(
                task_score_net, generated_params
            )

            # Enable grads for ISM loss computation (needs Jacobian)
            with torch.enable_grad():
                loss = self.ism_loss(task_score_net, task_data, task_masks)
                val_losses.append(loss.item())

        self.hypernetwork.train()
        return np.mean(val_losses)

    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        loss: float,
    ):
        """Save a training checkpoint.

        Args:
            path: File path for checkpoint.
            epoch: Current epoch number.
            loss: Current loss value.
        """
        checkpoint = {
            "epoch": epoch,
            "hypernetwork_state_dict": self.hypernetwork.state_dict(),
            "score_network_config": {
                "input_dim": self.score_network.input_dim,
                "hidden_dims": self.score_network.hidden_dims,
                "use_residual": self.score_network.use_residual,
            },
            "meta_optimizer_state_dict": self.meta_optimizer.state_dict(),
            "best_meta_loss": self.best_meta_loss,
            "loss": loss,
            "global_step": self.global_step,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load a training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.hypernetwork.load_state_dict(
            checkpoint["hypernetwork_state_dict"]
        )
        self.meta_optimizer.load_state_dict(
            checkpoint["meta_optimizer_state_dict"]
        )
        self.best_meta_loss = checkpoint.get("best_meta_loss", float("inf"))
        self.global_step = checkpoint.get("global_step", 0)

        if (
            self.scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.logger:
            self.logger.info(
                f"Loaded checkpoint from epoch {checkpoint['epoch']} "
                f"(loss: {checkpoint['loss']:.6f})"
            )
