from __future__ import annotations

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from vietrag.config import QwenConfig


class QwenClient:
    def __init__(self, config: QwenConfig):
        self.config = config
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = config.device_map or config.device or "auto"
        model_kwargs = {"trust_remote_code": True, "device_map": device_map, "torch_dtype": dtype}
        if config.quantization:
            if config.quantization == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                )
            else:
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=config.int8_cpu_offload,
                )
            model_kwargs["quantization_config"] = quant_config
            model_kwargs.pop("torch_dtype", None)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        self._input_device = config.device or _infer_primary_device(self.model)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self._input_device:
            inputs = {k: v.to(self._input_device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            min_p=self.config.min_p,
            presence_penalty=self.config.presence_penalty,
            do_sample=self.config.temperature > 0,
        )
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_length:]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return result.strip()


__all__ = ["QwenClient"]


def _infer_primary_device(model) -> Optional[str]:
    if hasattr(model, "device") and model.device:
        return str(model.device)
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        first_target = next(iter(device_map.values()))
        if isinstance(first_target, str):
            return first_target
        if isinstance(first_target, int):
            return f"cuda:{first_target}"
        if hasattr(first_target, "type"):
            return str(first_target)
    return None
