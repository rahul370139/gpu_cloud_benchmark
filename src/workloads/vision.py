"""ResNet-50 benchmark workload — synthetic ImageNet-shaped data."""

import torch
import torch.nn as nn
import torchvision.models as models

from .base import BaseWorkload, WorkloadMetadata


class ResNet50Workload(BaseWorkload):
    """ResNet-50 inference or training on synthetic (B, 3, 224, 224) tensors."""

    NUM_CLASSES = 1000

    def setup(self) -> None:
        self.model = models.resnet50(weights=None, num_classes=self.NUM_CLASSES)
        self.model.to(self.device)

        if self.mode == "training":
            self.model.train()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.model.eval()

    def generate_batch(self) -> dict[str, torch.Tensor]:
        images = torch.randn(self.batch_size, 3, 224, 224, device=self.device)
        batch = {"images": images}
        if self.mode == "training":
            batch["labels"] = torch.randint(0, self.NUM_CLASSES, (self.batch_size,), device=self.device)
        return batch

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch["images"])

    def _compute_loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "training":
            return self.criterion(output, batch["labels"])
        return output.mean()

    def get_metadata(self) -> WorkloadMetadata:
        param_count = sum(p.numel() for p in self.model.parameters())
        return WorkloadMetadata(
            name="resnet50",
            model_name="ResNet-50",
            param_count=param_count,
            input_shape=(self.batch_size, 3, 224, 224),
            throughput_unit="images/sec",
        )
