# server/answer.py
from __future__ import annotations
from dataclasses import dataclass
import re, torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass(frozen=True)
class AnswerConfig:
    model_name: str = "facebook/opt-125m"
    max_new_tokens: int = 64
    temperature: float = 0.0     # deterministic
    top_p: float = 1.0           # ignored when do_sample=False
    do_sample: bool = False
    repetition_penalty: float = 1.1


class AnswerEngine:
    def __init__(self, cfg: AnswerConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.model.eval()

        # OPT tokenizers sometimes have no pad_token set; safe default to eos.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _build_prompt(self, question: str) -> str:
        # A simple QA template that small base models handle better than “instructions”.
        question = question.strip()
        return (
            "Provide a factual answer in one sentence.\n"
            f"Q: {question}\n"
            "A:"
        )
    def answer(self, question: str, max_new_tokens: int | None = None) -> str:
        prompt = self._build_prompt(question)
        enc = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        max_new = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = out[0][enc["input_ids"].shape[1]:]
        ans = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        for stop in ["\nQ:", "\nQuestion:", " Q:", " Question:"]:
            if stop in ans:
                ans = ans.split(stop, 1)[0].strip()

        # cleanup
        ans = re.sub(r"\s+", " ", ans).strip()
        return ans

    def close(self) -> None:
        # Move model off GPU, drop references, and flush CUDA cache.
        try:
            self.model.to("cpu")
        except Exception:
            pass
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()