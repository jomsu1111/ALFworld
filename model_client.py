import difflib
import re
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _normalize_action(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


TASK_TYPES: Tuple[str, ...] = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)


TASK_FEW_SHOTS: Dict[str, str] = {

    "pick_and_place_simple": (
        "Task intent: pick an object and place it into a target receptacle.\n"
        "Example:\n"
        "Goal: put a apple in fridge.\n"
        "Observation: You are in the kitchen. On table 1, you see a apple 1. The fridge 1 is closed.\n"
        "Thought: The apple is on table 1. I should take it first.\n"
        "Action: take apple 1 from table 1\n"
    ),

    "look_at_obj_in_light": (
        "Task intent: examine an object under proper lighting.\n"
        "Example:\n"
        "Goal: examine a key under the lamp.\n"
        "Observation: On table 1, you see a key 1. The lamp 1 is off.\n"
        "Thought: I need the key in hand before examining it.\n"
        "Action: take key 1 from table 1\n"
    ),

    "pick_clean_then_place_in_recep": (
        "Task intent: clean an object before placing it into target receptacle.\n"
        "Example:\n"
        "Goal: put a clean mug in cabinet 1.\n"
        "Observation: On table 1, you see a mug 1. The sinkbasin 1 is empty.\n"
        "Thought: The mug must be cleaned first. I should take it.\n"
        "Action: take mug 1 from table 1\n"
    ),

    "pick_heat_then_place_in_recep": (
        "Task intent: heat an object before placing it into target receptacle.\n"
        "Example:\n"
        "Goal: put a hot soup in diningtable 1.\n"
        "Observation: On countertop 1, you see a soup 1. The microwave 1 is closed.\n"
        "Thought: I need to take the soup before heating it.\n"
        "Action: take soup 1 from countertop 1\n"
    ),

    "pick_cool_then_place_in_recep": (
        "Task intent: cool an object before placing it into target receptacle.\n"
        "Example:\n"
        "Goal: put a cool soda in cabinet 1.\n"
        "Observation: On table 1, you see a soda 1. The fridge 1 is closed.\n"
        "Thought: I should take the soda before cooling it.\n"
        "Action: take soda 1 from table 1\n"
    ),

    "pick_two_obj_and_place": (
        "Task intent: place two required objects into target receptacle.\n"
        "Example:\n"
        "Goal: put two apple in basket 1.\n"
        "Observation: On counter 1, you see a apple 1 and a apple 2.\n"
        "Thought: I should handle one apple at a time, starting with apple 1.\n"
        "Action: take apple 1 from counter 1\n"
    ),
}

PDDL_STYLE_GUIDE = (
    "ALFWorld / PDDL action constraints:\n"
    "1) Output exactly ONE admissible action string.\n"
    "2) Valid action formats are strictly:\n"
    "   - goto <recep>\n"
    "   - take <obj> from <recep>\n"
    "   - put <obj> in/on <recep>\n"
    "   - open <recep>\n"
    "   - close <recep>\n"
    "   - clean <obj> with <recep>\n"
    "   - heat <obj> with <recep>\n"
    "   - cool <obj> with <recep>\n"
    "   - toggle <obj>\n"
    "\n"
    "3) Preconditions:\n"
    "   - Must goto a receptacle before interacting with it.\n"
    "   - Must open a closed receptacle before take/put inside it.\n"
    "   - Must take an object before clean/heat/cool it.\n"
    "   - Must hold the object before put.\n"
    "\n"
    "4) Only hold ONE object at a time.\n"
    "5) If action fails (Nothing happens), reconsider location or preconditions.\n"
    "6) After clean/heat/cool, next goal is to place the object in target receptacle."
)


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
    prompting_mode: str
    use_reflexion: bool
    reflection_window: int


class LlamaActionPolicy:
    def __init__(self, cfg: PolicyConfig) -> None:
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.max_new_tokens = cfg.max_new_tokens
        self.history_window = cfg.history_window
        self.prompting_mode = cfg.prompting_mode
        self.use_reflexion = cfg.use_reflexion
        self.reflection_window = cfg.reflection_window

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
        task_type: str,
        reflections: Sequence[str],
    ) -> str:

        # ===== Recent trajectory =====
        history_slice = trajectory[-self.history_window :]
        history_lines = []
        for idx, (action, obs_text) in enumerate(history_slice, 1):
            history_lines.append(f"{idx}. action={action}")
            history_lines.append(f"   observation={obs_text}")
        history_text = "\n".join(history_lines) if history_lines else "(none)"

        # ===== Candidate actions =====
        candidates = "\n".join([f"{i + 1}. {cmd}" for i, cmd in enumerate(admissible_commands)])

        # ===== Reflexion (short + strong) =====
        reflection_text = ""
        if self.use_reflexion and reflections:
            recent_reflections = reflections[-self.reflection_window :]
            reflection_text = "Previous mistakes to avoid:\n"
            for item in recent_reflections:
                reflection_text += f"- {item}\n"
            reflection_text += "\nAvoid repeating these mistakes.\n\n"

        goal_text = "unknown"
        goal_match = re.search(
            r"your task is to:\s*(.+?)(?:[.\n]|$)",
            observation,
            flags=re.IGNORECASE,
        )
        if goal_match:
            goal_text = goal_match.group(1).strip()

        task_few_shot = TASK_FEW_SHOTS.get(task_type, TASK_FEW_SHOTS["pick_and_place_simple"])

        return (
            "You are an ALFWorld decision-making agent.\n"
            "Choose EXACTLY ONE valid action from the candidate list.\n\n"

            f"{PDDL_STYLE_GUIDE}\n\n"

            f"Task type: {task_type}\n"
            f"Task-specific example:\n{task_few_shot}\n\n"

            "Before acting, check carefully:\n"
            "- Am I at the correct location?\n"
            "- Is the receptacle open if required?\n"
            "- Am I holding the required object?\n"
            "- If heating/cooling/cleaning is done, should I now place the object?\n\n"

            f"{reflection_text}"

            f"Current observation:\n{observation}\n\n"
            f"Your goal is: {goal_text}\n"
            "Progress check: Did the last action move you closer to the goal? (yes/no)\n"
            "If no, choose a different action type than the previous one.\n\n"

            f"Recent trajectory:\n{history_text}\n\n"

            f"Candidate actions:\n{candidates}\n\n"

            "Output strictly in this format:\n"
            "Action: <one exact action from candidate list>"
        )

    def _choose_fallback(self, admissible_commands: Sequence[str]) -> str:
        for cmd in admissible_commands:
            if _normalize_action(cmd) == "look":
                return cmd
        return admissible_commands[0]

    def _extract_action_candidate(self, raw_text: str) -> str:
        # BeliefUpdate 블록의 "Inventory:"를 액션으로 오인하지 않도록
        # Action/Answer 라인만 우선적으로 추출한다.
        action_lines = re.findall(
            r"(?:^|\n)\s*(?:action|answer)\s*:\s*(.+)",
            raw_text,
            flags=re.IGNORECASE,
        )
        if action_lines:
            return action_lines[-1].strip().strip('"').strip("'")

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return ""
        return lines[-1].strip().strip('"').strip("'")

    def _match_action(self, raw_text: str, admissible_commands: Sequence[str]) -> str:
        cleaned = self._extract_action_candidate(raw_text)
        normalized_map = {_normalize_action(c): c for c in admissible_commands}

        norm = _normalize_action(cleaned)
        if norm in normalized_map:
            return normalized_map[norm]

        idx_match = re.search(r"^\s*(\d+)\s*[\).\s]?", cleaned)
        if idx_match:
            idx = int(idx_match.group(1)) - 1
            if 0 <= idx < len(admissible_commands):
                return admissible_commands[idx]

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
        task_type: str = "pick_and_place_simple",
        reflections: Sequence[str] = (),
    ) -> Tuple[str, str]:

        prompt = self.build_prompt(
            observation,
            admissible_commands,
            trajectory,
            task_type,
            reflections,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a precise ALFWorld agent. Think briefly and output ONE valid action.",
            },
            {"role": "user", "content": prompt},
        ]

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )

        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        # 🔥 Deterministic decoding
        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=min(self.max_new_tokens, 40),  # 제한
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        prompt_len = model_inputs["input_ids"].shape[-1]
        new_tokens = generated[0][prompt_len:]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 🔥 Action 라인 추출 강화
        action_line = ""

        if "Action:" in raw_text:
            action_line = raw_text.split("Action:")[-1].strip()
        else:
            # 혹시 모델이 Action 없이 바로 action을 쓴 경우
            action_line = raw_text.strip()

        # 정규화
        action_line = action_line.split("\n")[0].strip()
        action_line = action_line.rstrip(".")
        action_line = action_line.strip()

        # 🔥 exact match 우선
        matched_action = None
        for cmd in admissible_commands:
            if action_line == cmd:
                matched_action = cmd
                break

        # 🔥 substring match 보완
        if matched_action is None:
            for cmd in admissible_commands:
                if cmd in action_line:
                    matched_action = cmd
                    break

        # 🔥 최종 fallback 최소화
        if matched_action is None:
            matched_action = self._choose_fallback(admissible_commands)

        return matched_action, raw_text

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
    prompting_mode: str = "react",
    use_reflexion: bool = True,
    reflection_window: int = 4,
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
        prompting_mode=prompting_mode,
        use_reflexion=use_reflexion,
        reflection_window=reflection_window,
    )
    if reuse and cfg in _POLICY_CACHE:
        return _POLICY_CACHE[cfg]

    policy = LlamaActionPolicy(cfg)
    if reuse:
        _POLICY_CACHE[cfg] = policy
    return policy
