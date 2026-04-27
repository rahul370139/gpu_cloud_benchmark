"""Synthetic decoder-only LLM inference workload for GPU benchmarking."""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.workloads.base import BaseWorkload, WorkloadMetadata


class LlmTextGenerationWorkload(BaseWorkload):
    """T4-friendly autoregressive text generation benchmark.

    This workload avoids external model downloads by instantiating a small
    GPT-style decoder from config. Each benchmark iteration measures both:
    - prompt prefill over a fixed input sequence
    - token-by-token decode using KV cache
    """

    VOCAB_SIZE = 8192
    MAX_POSITIONS = 256
    PROMPT_LENGTH = 128
    GENERATION_LENGTH = 32
    HIDDEN_SIZE = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8

    def setup(self) -> None:
        if self.mode != "inference":
            raise ValueError("llm_text_generation currently supports inference mode only")

        config = GPT2Config(
            vocab_size=self.VOCAB_SIZE,
            n_positions=self.MAX_POSITIONS,
            n_ctx=self.MAX_POSITIONS,
            n_embd=self.HIDDEN_SIZE,
            n_layer=self.NUM_LAYERS,
            n_head=self.NUM_HEADS,
            bos_token_id=0,
            eos_token_id=1,
            use_cache=True,
        )
        self.model = GPT2LMHeadModel(config)
        self.model.to(self.device)
        self.model.eval()

    def generate_batch(self) -> dict[str, torch.Tensor]:
        input_ids = torch.randint(
            low=0,
            high=self.VOCAB_SIZE,
            size=(self.batch_size, self.PROMPT_LENGTH),
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.ones(
            (self.batch_size, self.PROMPT_LENGTH),
            device=self.device,
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            return_dict=True,
        )

        past_key_values = outputs.past_key_values
        attention_mask = batch["attention_mask"]
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens = [next_token]

        for _ in range(self.GENERATION_LENGTH - 1):
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((self.batch_size, 1), device=self.device, dtype=attention_mask.dtype),
                ],
                dim=1,
            )
            outputs = self.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)

        return torch.cat(generated_tokens, dim=1)

    def samples_per_batch(self) -> int:
        return self.batch_size * (self.PROMPT_LENGTH + self.GENERATION_LENGTH)

    def get_metadata(self) -> WorkloadMetadata:
        param_count = sum(p.numel() for p in self.model.parameters())
        return WorkloadMetadata(
            name="llm_text_generation",
            model_name="Synthetic GPT Decoder",
            param_count=param_count,
            input_shape=(self.batch_size, self.PROMPT_LENGTH),
            throughput_unit="tokens/sec",
        )
