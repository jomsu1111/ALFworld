import difflib
import re
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _normalize_action(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass(frozen=True)
class PolicyConfig:
    model_id: str
    hf_token: str
    device_map: str
    load_in_4bit: bool
    temperature: float
    top_p: float
    max_new_tokens: int
    history_window: int


class LlamaActionPolicy:
    def __init__(self, cfg: PolicyConfig) -> None:
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.max_new_tokens = cfg.max_new_tokens
        self.history_window = cfg.history_window

        quantization_config = None
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if cfg.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=cfg.hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            token=cfg.hf_token,
            torch_dtype=torch_dtype,
            device_map=cfg.device_map,
            quantization_config=quantization_config,
        )
        self.model.eval()

    def build_prompt(
        self,
        observation: str,
        admissible_commands: Sequence[str],
        trajectory: Sequence[Tuple[str, str]],
    ) -> str:
        history_slice = trajectory[-self.history_window :]
        history_lines = []
        for idx, (action, obs_text) in enumerate(history_slice, 1):
            history_lines.append(f"{idx}. action={action}")
            history_lines.append(f"   observation={obs_text}")
        history_text = "\n".join(history_lines) if history_lines else "(none)"

        candidates = "\n".join([f"{i + 1}. {cmd}" for i, cmd in enumerate(admissible_commands)])
        return (
            "You are an ALFWorld text-game agent. Choose the best next action to solve the task.\n"
            "You must output exactly one action from the candidate list and nothing else.\n\n"
            f"Current observation:\n{observation}\n\n"
            f"Recent trajectory:\n{history_text}\n\n"
            f"Candidate actions:\n{candidates}\n\n"
            "Output format: return only the action text."
        )

    def _choose_fallback(self, admissible_commands: Sequence[str]) -> str:
        for cmd in admissible_commands:
            if _normalize_action(cmd) == "look":
                return cmd
        return admissible_commands[0]

    def _match_action(self, raw_text: str, admissible_commands: Sequence[str]) -> str:
        cleaned = raw_text.strip().strip('"').strip("'")
        normalized_map = {_normalize_action(c): c for c in admissible_commands}

        label_match = re.search(r"(?:^|\n)\s*(?:action|answer)\s*:\s*(.+)", cleaned, flags=re.IGNORECASE)
        if label_match:
            cleaned = label_match.group(1).strip().strip('"').strip("'")

        norm = _normalize_action(cleaned)
        if norm in normalized_map:
            return normalized_map[norm]

        idx_match = re.search(r"^\s*(\d+)\s*[\).\s]?", cleaned)
        if idx_match:
            idx = int(idx_match.group(1)) - 1
            if 0 <= idx < len(admissible_commands):
                return admissible_commands[idx]

        for line in cleaned.splitlines():
            line_norm = _normalize_action(
                line.replace("Action:", "").replace("action:", "").strip().strip('"').strip("'")
            )
            if line_norm in normalized_map:
                return normalized_map[line_norm]

        for cmd in admissible_commands:
            if _normalize_action(cmd) in norm:
                return cmd

        near = difflib.get_close_matches(norm, list(normalized_map.keys()), n=1, cutoff=0.55)
        if near:
            return normalized_map[near[0]]

        return self._choose_fallback(admissible_commands)

    @torch.no_grad()
    def select_action(
        self,
        observation: str,
        admissible_commands: Sequence[str],
        trajectory: Sequence[Tuple[str, str]],
    ) -> Tuple[str, str]:
        prompt = self.build_prompt(observation, admissible_commands, trajectory)
        messages = [
            {"role": "system", "content": "You are a precise decision-making agent for ALFWorld."},
            {"role": "user", "content": prompt},
        ]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = model_inputs["input_ids"].shape[-1]
        new_tokens = generated[0][prompt_len:]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        action = self._match_action(raw_text, admissible_commands)
        return action, raw_text


_POLICY_CACHE: Dict[PolicyConfig, LlamaActionPolicy] = {}


def get_llama_action_policy(
    *,
    model_id: str,
    hf_token: str,
    device_map: str,
    load_in_4bit: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    history_window: int,
    reuse: bool = True,
) -> LlamaActionPolicy:
    cfg = PolicyConfig(
        model_id=model_id,
        hf_token=hf_token,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        history_window=history_window,
    )
    if reuse and cfg in _POLICY_CACHE:
        return _POLICY_CACHE[cfg]

    policy = LlamaActionPolicy(cfg)
    if reuse:
        _POLICY_CACHE[cfg] = policy
    return policy
