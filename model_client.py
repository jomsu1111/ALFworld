import re
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _normalize_action(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_goal_entities(goal_text: str) -> Tuple[str, str]:
    goal = goal_text.lower().strip()

    look_match = re.search(
        r"look at (?:a|an|the)?\s*(?P<obj>[a-z0-9_]+)\s+under (?:the\s+)?(?P<recep>[a-z0-9_]+)",
        goal,
    )
    if look_match:
        return look_match.group("obj"), look_match.group("recep")

    examine_match = re.search(
        r"examine (?:a|an|the)?\s*(?P<obj>[a-z0-9_]+)\s+with (?:the\s+)?(?P<recep>[a-z0-9_]+)",
        goal,
    )
    if examine_match:
        return examine_match.group("obj"), examine_match.group("recep")

    transform_match = re.search(
        r"(?:heat|cool|clean)\s+(?:some|a|an)?\s*(?P<obj>[a-z0-9_]+)\s+and put it\s+(?:in|on)\s+(?:the\s+)?(?P<recep>[a-z0-9_]+)",
        goal,
    )
    if transform_match:
        return transform_match.group("obj"), transform_match.group("recep")

    put_match = re.search(
        r"put (?:some|a|an|the|two)?\s*(?:(?:hot|cool|clean)\s+)?(?P<obj>[a-z0-9_]+)\s+(?:in|on)\s+(?:the\s+)?(?P<recep>[a-z0-9_]+)",
        goal,
    )
    if put_match:
        return put_match.group("obj"), put_match.group("recep")

    return "unknown", "unknown"


TASK_TYPES: Tuple[str, ...] = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)

MANDATORY_STEP_ORDER = {
    "pick_and_place_simple": (
        "Find target object",
        "Take target object",
        "Place target object in/on target receptacle",
    ),
    "pick_clean_then_place_in_recep": (
        "Find target object",
        "Take target object",
        "Clean target object with sinkbasin",
        "Place target object in/on target receptacle",
    ),
    "pick_heat_then_place_in_recep": (
        "Find target object",
        "Take target object",
        "Heat target object with microwave",
        "Place target object in/on target receptacle",
    ),
    "pick_cool_then_place_in_recep": (
        "Find target object",
        "Take target object",
        "Cool target object with fridge",
        "Place target object in/on target receptacle",
    ),
    "pick_two_obj_and_place": (
        "Find first target object",
        "Take first target object",
        "Place first target object in/on target receptacle",
        "Find second target object",
        "Take second target object",
        "Place second target object in/on target receptacle",
    ),
    "look_at_obj_in_light": (
        "Find target object",
        "Take target object",
        "Use target light source",
        "Look while holding target object",
    ),
}


def _contains_target_put(action: str, target_obj: str, target_recep: str) -> bool:
    action_l = action.strip().lower()
    if not action_l.startswith("put "):
        return False
    if target_obj != "unknown" and target_obj not in action_l:
        return False
    if target_recep != "unknown":
        return f"in/on {target_recep}" in action_l
    return True


def _is_stage_done(
    stage: str,
    trajectory: Sequence[Tuple[str, str]],
    target_obj: str,
    target_recep: str,
) -> bool:
    for action, _ in trajectory:
        action_l = action.strip().lower()

        if stage == "Find target object":
            if target_obj == "unknown":
                if action_l.startswith(("go to ", "open ", "examine ")):
                    return True
            elif target_obj in action_l:
                return True

        elif stage == "Take target object":
            if action_l.startswith("take ") and (target_obj == "unknown" or target_obj in action_l):
                return True

        elif stage == "Clean target object with sinkbasin":
            if action_l.startswith("clean ") and (target_obj == "unknown" or target_obj in action_l):
                return True

        elif stage == "Heat target object with microwave":
            if action_l.startswith("heat ") and (target_obj == "unknown" or target_obj in action_l):
                return True

        elif stage == "Cool target object with fridge":
            if action_l.startswith("cool ") and (target_obj == "unknown" or target_obj in action_l):
                return True

        elif stage == "Place target object in/on target receptacle":
            if _contains_target_put(action_l, target_obj, target_recep):
                return True

        elif stage == "Use target light source":
            if action_l.startswith("use ") and (target_recep == "unknown" or target_recep in action_l):
                return True

        elif stage == "Look while holding target object":
            if action_l == "look":
                return True

        elif stage == "Find first target object":
            if target_obj == "unknown":
                if action_l.startswith(("go to ", "open ", "examine ")):
                    return True
            elif target_obj in action_l:
                return True

        elif stage == "Take first target object":
            if action_l.startswith("take ") and (target_obj == "unknown" or target_obj in action_l):
                return True

        elif stage == "Place first target object in/on target receptacle":
            if _contains_target_put(action_l, target_obj, target_recep):
                return True

        elif stage == "Find second target object":
            if target_obj == "unknown":
                if action_l.startswith(("go to ", "open ", "examine ")):
                    return True
            elif target_obj in action_l:
                return True

        elif stage == "Take second target object":
            # Second object taken after first placement.
            if action_l.startswith("take ") and (target_obj == "unknown" or target_obj in action_l):
                first_placed = any(
                    _contains_target_put(prev_action, target_obj, target_recep)
                    for prev_action, _ in trajectory
                )
                if first_placed:
                    return True

        elif stage == "Place second target object in/on target receptacle":
            put_count = 0
            for prev_action, _ in trajectory:
                if _contains_target_put(prev_action, target_obj, target_recep):
                    put_count += 1
                    if put_count >= 2:
                        return True

    return False


def _next_required_stage(
    task_type: str,
    trajectory: Sequence[Tuple[str, str]],
    target_obj: str,
    target_recep: str,
) -> str:
    order = MANDATORY_STEP_ORDER.get(task_type, MANDATORY_STEP_ORDER["pick_and_place_simple"])
    for stage in order:
        if not _is_stage_done(stage, trajectory, target_obj, target_recep):
            return stage
    return "All mandatory steps are complete; finish with the most goal-consistent admissible action."


REACT_FEW_SHOTS_BY_TASK = {
    "pick_and_place_simple": (
        """react_put_short:
Your task is to: put some spraybottle on toilet.
> think: Find target object, take it, then place it at goal receptacle.
> go to cabinet 1
> open cabinet 1
> go to cabinet 2
> open cabinet 2
> take spraybottle 1 from cabinet 2
> go to toilet 1
> put spraybottle 1 in/on toilet 1
"""
    ),
    "pick_clean_then_place_in_recep": (
        """react_clean_short:
Your task is to: put a clean lettuce in diningtable.
> think: Find object -> take -> clean -> place.
> go to fridge 1
> open fridge 1
> go to diningtable 1
> take lettuce 1 from diningtable 1
> go to sinkbasin 1
> clean lettuce 1 with sinkbasin 1
> go to diningtable 1
> put lettuce 1 in/on diningtable 1
"""
    ),
    "pick_heat_then_place_in_recep": (
        """react_heat_short:
Your task is to: heat some egg and put it in diningtable.
> think: Find object -> take -> heat -> place.
> go to fridge 1
> open fridge 1
> go to countertop 3
> take egg 2 from countertop 3
> go to microwave 1
> heat egg 2 with microwave 1
> go to diningtable 1
> put egg 2 in/on diningtable 1
"""
    ),
    "pick_cool_then_place_in_recep": (
        """react_cool_short:
Your task is to: cool some pan and put it in stoveburner.
> think: Find object -> take -> cool -> place.
> go to stoveburner 1
> go to stoveburner 2
> go to stoveburner 3
> take pan 1 from stoveburner 3
> go to fridge 1
> cool pan 1 with fridge 1
> go to stoveburner 1
> put pan 1 in/on stoveburner 1
"""
    ),
    "pick_two_obj_and_place": (
        """react_puttwo_short:
Your task is to: put two creditcard in dresser.
> think: find first object -> take -> place -> find second object -> take -> place.
> go to drawer 1
> open drawer 1
> go to countertop 1
> take creditcard 2 from countertop 1
> go to dresser 1
> put creditcard 2 in/on dresser 1
> go to countertop 1
> take creditcard 3 from countertop 1
> go to dresser 1
> put creditcard 3 in/on dresser 1
"""
    ),
    "look_at_obj_in_light": (
        """react_examine_short:
Your task is to: look at bowl under the desklamp.
> think: Find bowl, hold it, then find and use lamp.
> go to shelf 2
> take bowl 1 from shelf 2
> go to desk 1
> go to sidetable 2
> use desklamp 1
> look
"""
    ),
}

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
    enforce_step_order: bool = True
    use_few_shot: bool = True
    use_react_format: bool = True


class LlamaActionPolicy:
    def __init__(self, cfg: PolicyConfig) -> None:
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.max_new_tokens = cfg.max_new_tokens
        self.history_window = cfg.history_window
        self.prompting_mode = cfg.prompting_mode
        self.enforce_step_order = cfg.enforce_step_order
        self.use_few_shot = cfg.use_few_shot
        self.use_react_format = cfg.use_react_format

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

    def _summarize_trajectory(self, trajectory: Sequence[Tuple[str, str]]) -> str:
        history_slice = trajectory[-self.history_window :]
        if not history_slice:
            return "Visited: (none)\nHolding: (none)\nOpened: (none)"

        visited = []
        opened = []
        holding = None

        for action, _ in history_slice:
            action_l = action.strip().lower()

            if action_l.startswith("go to "):
                place = action[len("go to ") :].strip()
                if place and place not in visited:
                    visited.append(place)
            elif action_l.startswith("open "):
                recep = action[len("open ") :].strip()
                if recep and recep not in opened:
                    opened.append(recep)
                if recep and recep not in visited:
                    visited.append(recep)
            elif action_l.startswith("take "):
                obj = action[len("take ") :].split(" from ", 1)[0].strip()
                holding = obj or holding
            elif action_l.startswith(("put ", "move ")):
                holding = None

        visited_text = ", ".join(visited) if visited else "(none)"
        opened_text = ", ".join(opened) if opened else "(none)"
        holding_text = holding if holding else "(none)"
        return f"Visited: {visited_text}\nHolding: {holding_text}\nOpened: {opened_text}"

    def build_prompt(
        self,
        observation: str,
        admissible_commands: Sequence[str],
        trajectory: Sequence[Tuple[str, str]],
        task_type: str,
        goal_text: str | None = None,
    ) -> str:
        state_summary = self._summarize_trajectory(trajectory)

        if not goal_text:
            parsed_goal = "unknown"
            goal_match = re.search(
                r"your task is to:?\s*(.+?)(?:[.\n]|$)",
                observation,
                flags=re.IGNORECASE,
            )
            if goal_match:
                parsed_goal = goal_match.group(1).strip()
            goal_text = parsed_goal
        few_shot = ""
        if self.use_few_shot:
            few_shot = REACT_FEW_SHOTS_BY_TASK.get(
                task_type,
                REACT_FEW_SHOTS_BY_TASK["pick_and_place_simple"],
            )
        target_obj, target_recep = _extract_goal_entities(goal_text)
        choices = "\n".join([f"{i + 1}. {cmd}" for i, cmd in enumerate(admissible_commands)])
        max_idx = len(admissible_commands)
        example_entities = (
            "egg, lettuce, pan, spraybottle, creditcard, bowl, diningtable, "
            "stoveburner, toilet, dresser, desklamp"
        )
        mandatory_step_block = ""
        if self.enforce_step_order:
            step_order = MANDATORY_STEP_ORDER.get(
                task_type,
                MANDATORY_STEP_ORDER["pick_and_place_simple"],
            )
            next_stage = _next_required_stage(task_type, trajectory, target_obj, target_recep)
            step_order_text = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(step_order)])
            mandatory_step_block = (
                "Mandatory step order (must follow strictly):\n"
                f"{step_order_text}\n"
                f"Current required step: {next_stage}\n"
            )
        mandatory_constraint_lines = ""
        if self.enforce_step_order:
            mandatory_constraint_lines = (
                "- Follow the mandatory step order exactly.\n"
                "- Never execute a later step before completing the current required step.\n"
                "- For clean/heat/cool tasks, do not place target object before applying clean/heat/cool.\n"
            )

        output_format = (
            "Thought: <1 concise sentence>\n"
            f"Action: <single integer between 1 and {max_idx}>"
            if self.use_react_format
            else f"Action: <single integer between 1 and {max_idx}>"
        )

        return (
            "You are an ALFWorld decision-making agent.\n"
            "You should imitate the decision pattern and reasoning process, but not imitate entities.\n"
            "Choose exactly one next action from admissible commands.\n"
            "Do not repeat a failed action unless there is new evidence.\n\n"

            f"Task type: {task_type}\n"
            f"{few_shot}\n\n"

            f"Current observation:\n{observation}\n\n"
            f"Your goal is: {goal_text}\n"
            f"Current target object: {target_obj}\n"
            f"Current target receptacle/tool: {target_recep}\n"
            f"{mandatory_step_block}"
            "Reasoning constraints:\n"
            "- Use few-shot for strategy only, never copy its entities.\n"
            "- In Thought, reference only current goal/admissible entities.\n"
            f"- Ignore example entities unless currently relevant: {example_entities}\n"
            f"{mandatory_constraint_lines}"

            f"State summary:\n{state_summary}\n\n"
            f"Admissible commands (numbered):\n{choices}\n\n"
            "Output format (must match):\n"
            f"{output_format}"
        )

    def _choose_fallback(self, admissible_commands: Sequence[str]) -> str:
        for cmd in admissible_commands:
            if _normalize_action(cmd) == "look":
                return cmd
        return admissible_commands[0]

    def _choose_non_repeating(
        self,
        admissible_commands: Sequence[str],
        blocked_norm: str,
    ) -> str:
        priority_prefixes = ("take ", "put ", "move ", "heat ", "cool ", "clean ", "use ")
        explore_prefixes = ("go to ", "open ")

        for prefix_group in (priority_prefixes, explore_prefixes):
            for cmd in admissible_commands:
                norm = _normalize_action(cmd)
                if norm == blocked_norm:
                    continue
                if any(norm.startswith(prefix) for prefix in prefix_group):
                    return cmd

        for cmd in admissible_commands:
            norm = _normalize_action(cmd)
            if norm != blocked_norm and norm != "look":
                return cmd

        return self._choose_fallback(admissible_commands)

    def _same_observation(self, a: str, b: str) -> bool:
        return re.sub(r"\s+", " ", a.strip().lower()) == re.sub(r"\s+", " ", b.strip().lower())

    def _repeat_streak(self, trajectory: Sequence[Tuple[str, str]], action_norm: str) -> int:
        streak = 0
        for past_action, _ in reversed(trajectory):
            if _normalize_action(past_action) == action_norm:
                streak += 1
            else:
                break
        return streak

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

    @torch.no_grad()
    def _generate_text(self, messages: Sequence[Dict[str, str]]) -> str:
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        generated = self.model.generate(**model_inputs, **gen_kwargs)
        prompt_len = model_inputs["input_ids"].shape[-1]
        new_tokens = generated[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _extract_thought(self, raw_text: str) -> str:
        thought_lines = re.findall(
            r"(?:^|\n)\s*thought\s*:\s*(.+)",
            raw_text,
            flags=re.IGNORECASE,
        )
        if thought_lines:
            return thought_lines[-1].strip().strip('"').strip("'")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if lines:
            return lines[0][:200]
        return "(no thought)"

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

        return self._choose_fallback(admissible_commands)

    @torch.no_grad()
    def select_action(
        self,
        observation: str,
        admissible_commands: Sequence[str],
        trajectory: Sequence[Tuple[str, str]],
        task_type: str = "pick_and_place_simple",
        goal_text: str | None = None,
    ) -> Tuple[str, str]:

        prompt = self.build_prompt(
            observation,
            admissible_commands,
            trajectory,
            task_type,
            goal_text,
        )

        messages = [
            {"role": "system", "content": "You are a precise ALFWorld agent."},
            {"role": "user", "content": prompt},
        ]
        raw_text = self._generate_text(messages)

        thought = self._extract_thought(raw_text)
        matched_action = self._match_action(raw_text, admissible_commands)

        # Guard against getting stuck: if model keeps proposing the same action
        # while observations are not changing, force a different admissible action.
        matched_norm = _normalize_action(matched_action)
        if trajectory:
            last_action_norm = _normalize_action(trajectory[-1][0])
            repeat_streak = self._repeat_streak(trajectory, matched_norm)
            no_new_evidence = (
                len(trajectory) >= 2
                and self._same_observation(trajectory[-1][1], trajectory[-2][1])
            )
            if matched_norm == last_action_norm and (repeat_streak >= 2 or no_new_evidence):
                matched_action = self._choose_non_repeating(admissible_commands, blocked_norm=matched_norm)

        return matched_action, thought

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
    enforce_step_order: bool = True,
    use_few_shot: bool = True,
    use_react_format: bool = True,
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
        enforce_step_order=enforce_step_order,
        use_few_shot=use_few_shot,
        use_react_format=use_react_format,
    )
    if reuse and cfg in _POLICY_CACHE:
        return _POLICY_CACHE[cfg]

    policy = LlamaActionPolicy(cfg)
    if reuse:
        _POLICY_CACHE[cfg] = policy
    return policy
