"""Template for adding a custom workload to the benchmark framework."""

import torch
import torch.nn as nn

from src.workloads.base import BaseWorkload, WorkloadMetadata


class CustomWorkloadTemplate(BaseWorkload):
    """Copy this file and replace the model/batch logic with your workload."""

    def setup(self) -> None:
        self.model = nn.Linear(1, 1).to(self.device)
        if self.mode == "training":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)

    def generate_batch(self) -> dict[str, torch.Tensor]:
        inputs = torch.randn(self.batch_size, 1, device=self.device)
        return {"inputs": inputs}

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch["inputs"])

    def get_metadata(self) -> WorkloadMetadata:
        return WorkloadMetadata(
            name="custom_template",
            model_name="Custom Template",
            param_count=0,
            input_shape=(self.batch_size, 1),
            throughput_unit="samples/sec",
        )
