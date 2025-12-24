from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vietrag.config import QwenConfig


class QwenClient:
    def __init__(self, config: QwenConfig):
        self.config = config
        model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.config.device:
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.temperature > 0,
        )
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_length:]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return result.strip()


__all__ = ["QwenClient"]
