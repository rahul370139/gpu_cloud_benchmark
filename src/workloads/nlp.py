"""BERT-base benchmark workload — synthetic token sequences."""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from .base import BaseWorkload, WorkloadMetadata


class BertBaseWorkload(BaseWorkload):
    """BERT-base encoder inference or training on synthetic (B, seq_len) token IDs."""

    VOCAB_SIZE = 30522
    SEQ_LEN = 512

    def setup(self) -> None:
        config = BertConfig()
        self.model = BertModel(config)
        self.model.to(self.device)

        if self.mode == "training":
            self.model.train()
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        else:
            self.model.eval()

    def generate_batch(self) -> dict[str, torch.Tensor]:
        input_ids = torch.randint(
            0, self.VOCAB_SIZE, (self.batch_size, self.SEQ_LEN), device=self.device,
        )
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return outputs.last_hidden_state

    def _compute_loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return output.mean()

    def samples_per_batch(self) -> int:
        """For NLP, report throughput in tokens/sec: batch_size * seq_len."""
        return self.batch_size * self.SEQ_LEN

    def get_metadata(self) -> WorkloadMetadata:
        param_count = sum(p.numel() for p in self.model.parameters())
        return WorkloadMetadata(
            name="bert_base",
            model_name="BERT-base-uncased",
            param_count=param_count,
            input_shape=(self.batch_size, self.SEQ_LEN),
            throughput_unit="tokens/sec",
        )
