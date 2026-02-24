import ast
import difflib
import re
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

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
        "Task intent: pick an object and place it into a target receptacle.\\n"
        "Example:\\n"
        "Observation: You need to put the apple in the fridge.\\n"
        "Reasoning: locate apple -> take apple -> locate fridge -> open if needed -> put.\\n"
        "Action: take apple from table"
    ),
    "look_at_obj_in_light": (
        "Task intent: examine an object under proper lighting.\\n"
        "Example:\\n"
        "Observation: You need to examine the key under the lamp.\\n"
        "Reasoning: find key -> take key -> locate lamp -> turn on lamp -> use examine/look.\\n"
        "Action: turn on lamp"
    ),
    "pick_clean_then_place_in_recep": (
        "Task intent: clean an object before placing it into target receptacle.\\n"
        "Example:\\n"
        "Observation: Put the clean mug in the cabinet.\\n"
        "Reasoning: find mug -> take mug -> find sink -> clean mug -> go to cabinet -> put mug.\\n"
        "Action: clean mug with sink"
    ),
    "pick_heat_then_place_in_recep": (
        "Task intent: heat an object before placing it into target receptacle.\\n"
        "Example:\\n"
        "Observation: Put the heated soup in the dining table.\\n"
        "Reasoning: find soup -> take soup -> find microwave/stove -> heat -> move -> put.\\n"
        "Action: heat soup with microwave"
    ),
    "pick_cool_then_place_in_recep": (
        "Task intent: cool an object before placing it into target receptacle.\\n"
        "Example:\\n"
        "Observation: Put the cooled soda in the bar cabinet.\\n"
        "Reasoning: find soda -> take soda -> find fridge -> cool/chill -> move -> put.\\n"
        "Action: cool soda with fridge"
    ),
    "pick_two_obj_and_place": (
        "Task intent: place two required objects into target receptacle.\\n"
        "Example:\\n"
        "Observation: Put two apples in the basket.\\n"
        "Reasoning: place first apple fully, then repeat for second apple.\\n"
        "Action: take apple from counter"
    ),
}


PDDL_STYLE_GUIDE = (
    "ALFWorld/PDDL-style action rules (practical):\\n"
    "1) You can only execute actions in the admissible command list.\\n"
    "2) Preconditions matter: usually open receptacles before put/take when required.\\n"
    "3) State-changing tasks require explicit verbs: clean/heat/cool/turn on.\\n"
    "4) For navigation bottlenecks, use goto/open/look to reveal missing affordances.\\n"
    "5) Prefer progress actions over repeats; if stuck, change subgoal (explore container/light source/appliance)."
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
        self.temperature = min(max(cfg.temperature, 0.0), 0.2)
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
        belief_state: Dict[str, Any],
    ) -> str:
        history_slice = trajectory[-self.history_window :]
        history_lines = []
        for idx, (action, obs_text) in enumerate(history_slice, 1):
            history_lines.append(f"{idx}. action={action}")
            history_lines.append(f"   observation={obs_text}")
        history_text = "\n".join(history_lines) if history_lines else "(none)"

        candidates = "\n".join([f"{i + 1}. {cmd}" for i, cmd in enumerate(admissible_commands)])
        task_example = TASK_FEW_SHOTS.get(task_type, "")

        reflection_text = "(none)"
        if self.use_reflexion and reflections:
            reflection_text = "\n".join([f"- {item}" for item in reflections[-self.reflection_window :]])

        if self.prompting_mode == "react":
            format_rules = (
                "Output format strictly:\n"
                "BeliefUpdate:\n"
                "  CompletedSubgoals: [ ... ]\n"
                "  Inventory: [ ... ]\n"
                "  ObjectStates: { ... }\n"
                "  LoopCounter: <int>\n"
                "Reasoning:\n"
                "  Thought: <short reasoning about next subgoal and preconditions>\n"
                "Action: <one exact action from candidate actions>"
            )
        else:
            format_rules = (
                "Output format strictly:\n"
                "BeliefUpdate:\n"
                "  CompletedSubgoals: [ ... ]\n"
                "  Inventory: [ ... ]\n"
                "  ObjectStates: { ... }\n"
                "  LoopCounter: <int>\n"
                "Reasoning:\n"
                "  Thought: <short reasoning>\n"
                "Action: <one exact action from candidate actions>"
            )

        return (
            "You are an ALFWorld text-game agent. Choose the best next action to solve the task.\n"
            "Act with explicit precondition-aware planning and choose only from candidate actions.\n\n"
            f"Detected task type: {task_type}\n\n"
            "Task taxonomy (6 tasks):\n"
            "- pick_and_place_simple\n"
            "- look_at_obj_in_light\n"
            "- pick_clean_then_place_in_recep\n"
            "- pick_heat_then_place_in_recep\n"
            "- pick_cool_then_place_in_recep\n"
            "- pick_two_obj_and_place\n\n"
            f"Task-specific few-shot example:\n{task_example}\n\n"
            f"{PDDL_STYLE_GUIDE}\n\n"
            "Subgoal transition rule:\n"
            "- If a subgoal is completed, explicitly mark it in CompletedSubgoals and choose the next subgoal.\n"
            "- After heating/cooling/cleaning an object, you must transition to placing it in the target receptacle.\n\n"
            f"Reflexion memory from failed attempts:\n{reflection_text}\n\n"
            "Current structured belief state:\n"
            f"BeliefState:\n"
            f"  CompletedSubgoals: {belief_state.get('CompletedSubgoals', [])}\n"
            f"  Inventory: {belief_state.get('Inventory', [])}\n"
            f"  ObjectStates: {belief_state.get('ObjectStates', {})}\n"
            f"  LoopCounter: {belief_state.get('LoopCounter', 0)}\n\n"
            f"Current observation:\n{observation}\n\n"
            f"Recent trajectory:\n{history_text}\n\n"
            f"Candidate actions:\n{candidates}\n\n"
            f"{format_rules}"
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

    def _extract_belief_update(self, raw_text: str, prev_belief: Dict[str, Any]) -> Dict[str, Any]:
        belief = {
            "CompletedSubgoals": list(prev_belief.get("CompletedSubgoals", [])),
            "Inventory": list(prev_belief.get("Inventory", [])),
            "ObjectStates": dict(prev_belief.get("ObjectStates", {})),
            "LoopCounter": int(prev_belief.get("LoopCounter", 0)),
        }
        patterns = {
            "CompletedSubgoals": r"CompletedSubgoals\s*:\s*(\[[^\n]*\])",
            "Inventory": r"Inventory\s*:\s*(\[[^\n]*\])",
            "ObjectStates": r"ObjectStates\s*:\s*(\{[^\n]*\})",
            "LoopCounter": r"LoopCounter\s*:\s*(-?\d+)",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, raw_text, flags=re.IGNORECASE)
            if not m:
                continue
            value = m.group(1).strip()
            if key in {"CompletedSubgoals", "Inventory", "ObjectStates"}:
                try:
                    parsed = ast.literal_eval(value)
                    if key != "ObjectStates" and isinstance(parsed, list):
                        belief[key] = [str(x) for x in parsed]
                    if key == "ObjectStates" and isinstance(parsed, dict):
                        belief[key] = {str(k): str(v) for k, v in parsed.items()}
                except Exception:
                    continue
            else:
                try:
                    belief[key] = max(0, int(value))
                except ValueError:
                    continue
        return belief

    @torch.no_grad()
    def select_action(
        self,
        observation: str,
        admissible_commands: Sequence[str],
        trajectory: Sequence[Tuple[str, str]],
        task_type: str,
        reflections: Sequence[str],
        belief_state: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, Any]]:
        prompt = self.build_prompt(
            observation,
            admissible_commands,
            trajectory,
            task_type,
            reflections,
            belief_state,
        )
        messages = [
            {
                "role": "system",
                "content": "You are a precise decision-making ALFWorld agent using ReAct-style planning.",
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
        belief_update = self._extract_belief_update(raw_text, belief_state)
        action = self._match_action(raw_text, admissible_commands)
        if action not in admissible_commands:
            action = self._choose_fallback(admissible_commands)
        return action, raw_text, belief_update


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
