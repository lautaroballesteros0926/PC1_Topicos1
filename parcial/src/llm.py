"""LLM local del Experimento 2: Qwen2.5-0.5B-Instruct (el mismo modelo de la PC1 y la PC2), decoding greedy."""
from __future__ import annotations

import time
from typing import Any

LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DECODING = {"do_sample": False, "max_new_tokens": 200, "repetition_penalty": 1.05}


class LocalLLM:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(LLM_ID, revision=LLM_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(LLM_ID, revision=LLM_REVISION, dtype=torch.float32).eval()
        self.context_window = int(getattr(self.model.config, "max_position_embeddings", 0))

    def render(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def chat(self, system: str, user: str, max_new_tokens: int | None = None) -> dict[str, Any]:
        prompt = self.render(system, user)
        inputs = self.tok(prompt, return_tensors="pt")
        n_prompt = int(inputs["input_ids"].shape[1])
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens or DECODING["max_new_tokens"],
                                      repetition_penalty=DECODING["repetition_penalty"], temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.tok.eos_token_id)
        new = out[0, n_prompt:]
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        return {"text": text, "n_prompt_tokens": n_prompt, "n_new_tokens": int(new.shape[0]),
                "hit_max_new_tokens": int(new.shape[0]) >= (max_new_tokens or DECODING["max_new_tokens"]),
                "latency_s": round(time.perf_counter() - t0, 2), "prompt": prompt}
