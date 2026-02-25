import difflib
import re
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple


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


REACT_FEW_SHOTS = (
    "Few-shot Example 1 (pick-and-place):\n"
    "Goal: put a candle in bathtubbasin 1\n"
    "Observation: On countertop 1, you see a candle 1. bathtubbasin 1 is in the bathroom.\n"
    "Thought: I observed candle 1 on countertop 1, and the goal is to put it in bathtubbasin 1. "
    "To satisfy that goal, I must first hold candle 1 before moving to the target receptacle.\n"
    "Action: take candle 1 from countertop 1\n"
    "Observation: You are holding candle 1.\n"
    "Thought: I am now holding the goal object. Since the goal receptacle is bathtubbasin 1, "
    "the next required subgoal is to go to bathtubbasin 1.\n"
    "Action: go to bathtubbasin 1\n"
    "Observation: You are near bathtubbasin 1.\n"
    "Thought: I am at the target receptacle while holding the goal object, so placing it here directly completes the goal.\n"
    "Action: put candle 1 in/on bathtubbasin 1\n\n"
    "Few-shot Example 2 (heat-then-place):\n"
    "Goal: put a hot soup in diningtable 1\n"
    "Observation: On countertop 1, you see soup 1.\n"
    "Thought: I observed soup 1 on countertop 1, and the goal requires a hot soup in diningtable 1. "
    "To make progress, I must take soup 1 first so heating becomes possible.\n"
    "Action: take soup 1 from countertop 1\n"
    "Observation: You are holding soup 1. microwave 1 is closed.\n"
    "Thought: The goal needs the soup heated, but microwave 1 is closed. "
    "The immediate subgoal is to open microwave 1 so heating can be executed.\n"
    "Action: open microwave 1\n"
    "Observation: microwave 1 is open.\n"
    "Thought: Now the heating precondition is satisfied. I should heat soup 1 first, then later move to diningtable 1 and place it.\n"
    "Action: heat soup 1 with microwave 1\n"
)

PDDL_STYLE_GUIDE = (
    "ALFWorld / PDDL action constraints:\n"
    "1) Output exactly ONE admissible action string.\n"
    "2) Valid action formats are strictly:\n"
    "   - go to <recep>\n"
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
    "   - Must go to a receptacle before interacting with it.\n"
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
    api_key: str
    temperature: float
    top_p: float
    max_new_tokens: int
    history_window: int
    prompting_mode: str
    use_reflexion: bool
    reflection_window: int


class GPT5ActionPolicy:
    def __init__(self, cfg: PolicyConfig) -> None:
        self.model_id = cfg.model_id
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.max_new_tokens = cfg.max_new_tokens
        self.history_window = cfg.history_window
        self.prompting_mode = cfg.prompting_mode
        self.use_reflexion = cfg.use_reflexion
        self.reflection_window = cfg.reflection_window
        try:
            from openai import OpenAI
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "openai package is required. Install with: pip install openai"
            ) from e
        self.client = OpenAI(api_key=cfg.api_key)
        self.last_inferred_subgoal = ""
        self.step_index = 0

    def build_prompt(
        self,
        observation: str,
        admissible_commands: Sequence[str],
        trajectory: Sequence[Tuple[str, str]],
        task_type: str,
        reflections: Sequence[str],
    ) -> str:
        history_slice = trajectory[-self.history_window :]
        history_lines = []
        for idx, (action, obs_text) in enumerate(history_slice, 1):
            history_lines.append(f"{idx}. action={action}")
            history_lines.append(f"   observation={obs_text}")
        history_text = "\n".join(history_lines) if history_lines else "(none)"

        candidates = "\n".join([f"{i + 1}. {cmd}" for i, cmd in enumerate(admissible_commands)])

        reflection_text = ""
        if self.use_reflexion and reflections:
            recent_reflections = reflections[-self.reflection_window :]
            reflection_text = "Previous mistakes to avoid:\n"
            for item in recent_reflections:
                reflection_text += f"- {item}\n"
            reflection_text += "\nAvoid repeating these mistakes.\n\n"

        goal_text = "unknown"
        goal_match = re.search(
            r"your task is to:?\s*(.+?)(?:[.\n]|$)",
            observation,
            flags=re.IGNORECASE,
        )
        if goal_match:
            goal_text = goal_match.group(1).strip()
        subgoal_text = "unknown"
        subgoal_match = re.search(
            r"current subgoal:\s*(.+?)(?:\n|$)",
            observation,
            flags=re.IGNORECASE,
        )
        if subgoal_match:
            subgoal_text = subgoal_match.group(1).strip()
        step_no = self.step_index + 1
        last_subgoal = self.last_inferred_subgoal if self.last_inferred_subgoal else "(none)"

        return (
            "You are an ALFWorld decision-making agent.\n"
            "Use ReAct style: Observation -> Thought -> Action.\n"
            "Choose EXACTLY ONE valid action from the candidate list.\n\n"
            f"{PDDL_STYLE_GUIDE}\n\n"
            f"Task type: {task_type}\n"
            f"{REACT_FEW_SHOTS}\n"
            "Before acting, check carefully:\n"
            "- Am I at the correct location?\n"
            "- Is the receptacle open if required?\n"
            "- Am I holding the required object?\n"
            "- If heating/cooling/cleaning is done, should I now place the object?\n\n"
            "Thought quality rule:\n"
            "- Mention what you observed.\n"
            "- Mention the current subgoal.\n"
            "- State the immediate subgoal and why this action advances it.\n\n"
            "Per-step decision rule:\n"
            "- At every step, restate the task you are solving.\n"
            "- Infer ONE immediate subgoal for this step.\n"
            "- Choose ONE action that directly advances that subgoal.\n\n"
            "Step record:\n"
            f"- Step: {step_no}\n"
            f"- Task type: {task_type}\n"
            f"- Goal: {goal_text}\n"
            f"- Current subgoal hint: {subgoal_text}\n"
            f"- Last inferred subgoal: {last_subgoal}\n\n"
            f"{reflection_text}"
            f"Current observation:\n{observation}\n\n"
            f"Your goal is: {goal_text}\n"
            f"Current subgoal is: {subgoal_text}\n"
            "Progress check: Did the last action move you closer to the goal? (yes/no)\n"
            "If no, choose a different action type than the previous one.\n\n"
            f"Recent trajectory:\n{history_text}\n\n"
            f"Candidate actions:\n{candidates}\n\n"
            "Output strictly in this format:\n"
            "Subgoal: <one immediate subgoal for this step>\n"
            "Thought: <observation + goal + immediate subgoal reasoning>\n"
            "Action: <one exact action from candidate list>"
        )

    def _choose_fallback(self, admissible_commands: Sequence[str]) -> str:
        for cmd in admissible_commands:
            if _normalize_action(cmd) == "look":
                return cmd
        return admissible_commands[0]

    def _extract_action_candidate(self, raw_text: str) -> str:
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

    def _extract_subgoal(self, raw_text: str) -> str:
        subgoal_lines = re.findall(
            r"(?:^|\n)\s*subgoal\s*:\s*(.+)",
            raw_text,
            flags=re.IGNORECASE,
        )
        if subgoal_lines:
            return subgoal_lines[-1].strip().strip('"').strip("'")

        heuristic = re.search(
            r"(?:immediate|next)\s+subgoal\s+(?:is|:)\s*(.+?)(?:[.\n]|$)",
            raw_text,
            flags=re.IGNORECASE,
        )
        if heuristic:
            return heuristic.group(1).strip().strip('"').strip("'")
        return ""

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

    def _chat(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise ALFWorld agent. Think briefly and output ONE valid action.",
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=self.max_new_tokens,
        )
        return (completion.choices[0].message.content or "").strip()

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
        raw_text = self._chat(prompt)
        self.last_inferred_subgoal = self._extract_subgoal(raw_text)
        self.step_index += 1
        thought = self._extract_thought(raw_text)
        matched_action = self._match_action(raw_text, admissible_commands)
        return matched_action, thought


_POLICY_CACHE: Dict[PolicyConfig, GPT5ActionPolicy] = {}


def get_gpt5_action_policy(
    *,
    model_id: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    history_window: int,
    prompting_mode: str = "react",
    use_reflexion: bool = True,
    reflection_window: int = 4,
    reuse: bool = True,
) -> GPT5ActionPolicy:
    cfg = PolicyConfig(
        model_id=model_id,
        api_key=api_key,
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

    policy = GPT5ActionPolicy(cfg)
    if reuse:
        _POLICY_CACHE[cfg] = policy
    return policy
