"""Abstract base class for all benchmark workloads."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class WorkloadMetadata:
    name: str
    model_name: str
    param_count: int
    input_shape: tuple
    throughput_unit: str  # "images/sec", "tokens/sec", etc.


class BaseWorkload(ABC):
    """Contract every benchmark workload must satisfy."""

    def __init__(self, batch_size: int, device: str = "cuda", mode: str = "inference"):
        if mode not in ("inference", "training"):
            raise ValueError(f"mode must be 'inference' or 'training', got '{mode}'")
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.mode = mode
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    @abstractmethod
    def setup(self) -> None:
        """Load model, move to device, set eval/train mode, create optimizer if training."""

    @abstractmethod
    def generate_batch(self) -> dict[str, torch.Tensor]:
        """Return a dict of input tensors on *self.device* with the current batch_size."""

    @abstractmethod
    def get_metadata(self) -> WorkloadMetadata:
        """Return static metadata about this workload."""

    def warmup(self, n_iters: int = 10) -> None:
        """Run *n_iters* forward passes (discarded) to warm JIT caches and allocators."""
        batch = self.generate_batch()
        for _ in range(n_iters):
            self.run_iteration(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def run_iteration(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute one forward pass (+ backward + optimizer step if training).

        Returns the loss tensor (training) or output tensor (inference).
        """
        if self.mode == "inference":
            with torch.no_grad():
                output = self._forward(batch)
            return output

        output = self._forward(batch)
        loss = self._compute_loss(output, batch)
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    @abstractmethod
    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Model forward pass — implemented by each workload."""

    def _compute_loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Default loss: mean of output. Override for workload-specific loss."""
        return output.mean()

    def samples_per_batch(self) -> int:
        """Number of logical samples in one batch (for throughput calculation)."""
        return self.batch_size

    def cleanup(self) -> None:
        """Release GPU memory."""
        del self.model
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
