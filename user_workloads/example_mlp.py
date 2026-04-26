"""Example custom workload that users can copy and adapt."""

import torch
import torch.nn as nn

from src.workloads.base import BaseWorkload, WorkloadMetadata


class ExampleMLPWorkload(BaseWorkload):
    """Simple synthetic MLP used as a reference custom workload."""

    INPUT_DIM = 128
    HIDDEN_DIM = 256
    OUTPUT_DIM = 4

    def setup(self) -> None:
        self.model = nn.Sequential(
            nn.Linear(self.INPUT_DIM, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_DIM, self.OUTPUT_DIM),
        )
        self.model.to(self.device)
        if self.mode == "training":
            self.model.train()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.model.eval()

    def generate_batch(self) -> dict[str, torch.Tensor]:
        features = torch.randn(self.batch_size, self.INPUT_DIM, device=self.device)
        batch = {"features": features}
        if self.mode == "training":
            batch["labels"] = torch.randint(0, self.OUTPUT_DIM, (self.batch_size,), device=self.device)
        return batch

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch["features"])

    def _compute_loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "training":
            return self.criterion(output, batch["labels"])
        return output.mean()

    def get_metadata(self) -> WorkloadMetadata:
        param_count = sum(p.numel() for p in self.model.parameters())
        return WorkloadMetadata(
            name="example_mlp",
            model_name="Example-MLP",
            param_count=param_count,
            input_shape=(self.batch_size, self.INPUT_DIM),
            throughput_unit="samples/sec",
        )
