from typing import Optional

from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmupScheduler:
    """
    Implements learning rate warmup followed by ReduceLROnPlateau scheduling.
    During warmup, the learning rate linearly increases from 0 to the base learning rate.
    After warmup, ReduceLROnPlateau takes over for learning rate adjustment.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: Optional[int],
        base_lr: float,
        factor=0.5,
        patience=5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0

        # Initialize ReduceLROnPlateau scheduler for after warmup
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience
        )

        # Set initial learning rate to 0 if using warmup
        for param_group in self.optimizer.param_groups:
            if self.warmup_steps is None:
                param_group["lr"] = self.base_lr
            else:
                param_group["lr"] = 0.0

    def step(self, metrics=None):
        """
        Update learning rate based on current step.
        During warmup, increases linearly from 0 to base_lr.
        After warmup, delegates to ReduceLROnPlateau.

        Args:
            metrics: Validation metrics for ReduceLROnPlateau (only used post-warmup)
        """
        self.current_step += 1

        if self.warmup_steps is not None and self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            # After warmup, let ReduceLROnPlateau handle scheduling
            if metrics is not None:
                self.plateau_scheduler.step(metrics)

    def get_last_lr(self):
        """Returns current learning rate."""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]

    def state_dict(self):
        """Returns scheduler state dict for checkpointing."""
        return {
            "warmup_steps": self.warmup_steps,
            "base_lr": self.base_lr,
            "current_step": self.current_step,
            "plateau_scheduler": self.plateau_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Loads scheduler state from checkpoint."""
        self.warmup_steps = state_dict["warmup_steps"]
        self.base_lr = state_dict["base_lr"]
        self.current_step = state_dict["current_step"]
        self.plateau_scheduler.load_state_dict(state_dict["plateau_scheduler"])
