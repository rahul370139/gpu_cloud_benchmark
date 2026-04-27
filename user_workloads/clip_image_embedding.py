"""CLIP image embedding workload for benchmarking vision encoder inference."""

import torch
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

from src.workloads.base import BaseWorkload, WorkloadMetadata


class ClipImageEmbeddingWorkload(BaseWorkload):
    """Synthetic CLIP image embedding workload using the vision encoder only."""

    IMAGE_SIZE = 224
    CHANNELS = 3

    def setup(self) -> None:
        config = CLIPVisionConfig(
            image_size=self.IMAGE_SIZE,
            projection_dim=512,
        )
        self.model = CLIPVisionModelWithProjection(config)
        self.model.to(self.device)

        if self.mode == "training":
            self.model.train()
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        else:
            self.model.eval()

    def generate_batch(self) -> dict[str, torch.Tensor]:
        pixel_values = torch.randn(
            self.batch_size,
            self.CHANNELS,
            self.IMAGE_SIZE,
            self.IMAGE_SIZE,
            device=self.device,
        )
        return {"pixel_values": pixel_values}

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(pixel_values=batch["pixel_values"])
        image_embeds = outputs.image_embeds
        return torch.nn.functional.normalize(image_embeds, dim=-1)

    def _compute_loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return output.mean()

    def get_metadata(self) -> WorkloadMetadata:
        param_count = sum(p.numel() for p in self.model.parameters())
        return WorkloadMetadata(
            name="clip_image_embedding",
            model_name="CLIP-ViT Image Encoder",
            param_count=param_count,
            input_shape=(self.batch_size, self.CHANNELS, self.IMAGE_SIZE, self.IMAGE_SIZE),
            throughput_unit="images/sec",
        )
