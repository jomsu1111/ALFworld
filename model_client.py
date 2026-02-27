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


REACT_FEW_SHOTS_BY_TASK = {
    "pick_and_place_simple": (
        """react_put_0:
You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Your task is to: put some spraybottle on toilet.
> think: To solve the task, I need to find and take a sparybottle, then put it on toilet.
OK.
> think: First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.
OK.
> go to cabinet 1
On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.
> go to cabinet 2
The cabinet 2 is closed.
> open cabinet 2
You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.
> think: Now I find a spraybottle (2). Next, I need to take it.
> take spraybottle 2 from cabinet 2
You pick up the spraybottle 2 from the cabinet 2.
> think: Now I take a spraybottle (2). Next, I need to put it in/on toilet 1.
OK.
> go to toilet 1
On the toilet 1, you see a soapbottle 2.
> put spraybottle 2 in/on toilet 1
You put the spraybottle 2 in/on the toilet 1.
"""
    ),
    "pick_clean_then_place_in_recep": (
        """react_clean_0:
You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: put a clean lettuce in diningtable.
> think: To solve the task, I need to find and take a lettuce, then clean it with sinkbasin, then put it in diningtable. 
OK.
> think: First I need to find a lettuce. A lettuce is more likely to appear in fridge (1), diningtable (1), sinkbasin (1), stoveburner (1-3), cabinet (1-13). I can check one by one, starting with fridge 1.
OK.
> go to fridge 1
The fridge 1 is closed.
> open fridge 1
You open the fridge 1. The fridge 1 is open. In it, you see a cup 3, a egg 2, a potato 3, and a potato 2.
> go to diningtable 1
On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a lettuce 1, a mug 2, a mug 1, a pan 2, a peppershaker 1, a spatula 3, a tomato 2, and a tomato 1.
> think: Now I find a lettuce (1). Next, I need to take it.
OK.
> take lettuce 1 from diningtable 1
You pick up the lettuce 1 from the diningtable 1.
> think: Now I take a lettuce (1). Next, I need to go to sinkbasin (1) and clean it.
OK.
> go to sinkbasin 1
On the sinkbasin 1, you see a apple 2, a ladle 2, a spoon 1, and a tomato 3.
> clean lettuce 1 with sinkbasin 1
You clean the lettuce 1 using the sinkbasin 1.
> think: Now I clean a lettuce (1). Next, I need to put it in/on diningtable 1.
OK.
> go to diningtable 1
On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a mug 2, a mug 1, a pan 2, a peppershaker 1, a spatula 3, a tomato 2, and a tomato 1.
> put lettuce 1 in/on diningtable 1
You put the lettuce 1 in/on the diningtable 1.
"""
    ),
    "pick_heat_then_place_in_recep": (
        """react_heat_0:
You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a diningtable 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: heat some egg and put it in diningtable.
> think: To solve the task, I need to find and take an egg, then heat it with microwave, then put it in diningtable.
OK.
> think: First I need to find an egg. An egg is more likely to appear in fridge (1), countertop (1-3), diningtable (1), stoveburner (1-4), toaster (1), garbagecan (1), cabinet (1-10). I can check one by one, starting with fridge 1.
OK.
> open fridge 1
You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, and a potato 3.
> go to countertop 1
On the countertop 1, you see a bread 1, a fork 1, and a saltshaker 1.
> go to countertop 2
On the countertop 2, you see nothing.
> go to countertop 3
On the countertop 3, you see a bowl 1, a butterknife 1, a egg 2, a kettle 2, a plate 1, a sink 1, and a spatula 2.
> think: Now I find an egg (2). Next, I need to take it.
OK.
> take egg 2 from countertop 3
You pick up the egg 2 from the countertop 3.
> think: Now I take an egg (2). Next, I need go to a microwave (1) and heat it.
OK.
> go to microwave 1
The microwave 1 is closed.
> heat egg 2 with microwave 1
You heat the egg 2 using the microwave 1.
> think: Now I heat an egg (2). Next, I need to put it in/on diningtable 1.
OK.
> go to diningtable 1
On the diningtable 1, you see a apple 2, a bread 3, a egg 1, a kettle 1, a knife 1, a mug 1, a papertowelroll 1, a peppershaker 2, a potato 1, a soapbottle 1, and a spatula 1.
> put egg 2 in/on diningtable 1
You put the egg 2 in/on the diningtable 1.
"""
    ),
    "pick_cool_then_place_in_recep": (
        """react_cool_0:
You are in the middle of a room. Looking quickly around you, you see a cabinet 16, a cabinet 15, a cabinet 14, a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a safe 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: cool some pan and put it in stoveburner.
> think: To solve the task, I need to find and take a pan, then cool it with fridge, then put it in stoveburner.
OK.
> think: First I need to find a pan. An pan is more likely to appear in stoveburner (1-4), sinkbasin (1), diningtable (1), countertop (1-2), cabinet (1-16), drawer (1-5). I can check one by one, starting with stoveburner 1.
OK.
>go to stoveburner 1
On the stoveburner 1, you see nothing.
>go to stoveburner 2
On the stoveburner 2, you see a pot 1.
> go to stoveburner 3
On the stoveburner 3, you see a pan 1.
> take pan 1 from stoveburner 3
> think: Now I find a pan (1). Next, I need to take it.
OK.
You pick up the pan 1 from the stoveburner 3.
> think: Now I take a pan (1). Next, I need to go to a fridge (1) and cool it.
OK.
> go to fridge 1
The fridge 1 is closed.
> cool pan 1 with fridge 1
You cool the pan 1 using the fridge 1.
> think: Now I cool a pan (1). Next, I need to put it in/on stoveburner 1.
OK.
>go to stoveburner 1
On the stoveburner 1, you see nothing.
> put pan 1 in/on stoveburner 1
You put the pan 1 in/on the stoveburner 1.
"""
    ),
    "pick_two_obj_and_place": (
        """react_puttwo_0:
You are in the middle of a room. Looking quickly around you, you see a armchair 2, a armchair 1, a bed 1, a countertop 1, a diningtable 1, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a laundryhamper 1, and a sidetable 1.
Your task is to: put two creditcard in dresser.
> think: To solve the task, I need to find and take the first creditcard, then put it in dresser, then find and take the second creditcard, then put it in dresser.
OK.
> think: First I need to find the first creditcard. A creditcard is more likely to appear in drawer (1-2), coutertop (1), sidetable (1), diningtable (1), armchair (1-2), bed (1). I can check one by one, starting with drawer 1.
OK.
> go to drawer 1
The drawer 1 is closed.
> open drawer 1
You open the drawer 1. The drawer 1 is open. In it, you see a book 1, a cd 1, and a pen 1.
> go to drawer 2
The drawer 2 is closed.
> open drawer 2
You open the drawer 2. The drawer 2 is open. In it, you see nothing.
> go to countertop 1
On the countertop 1, you see a cellphone 2, a creditcard 4, a creditcard 3, a creditcard 2, a mirror 1, a pencil 2, and a pencil 1.
> think: Now I find the first creditcard (2). Next, I need to take it. I can find the second creditcard (3) later in countertop 1.
OK.
> take creditcard 2 from countertop 1
You pick up the creditcard 2 from the countertop 1.
> think: Now I take the first creditcard (2). Next, I need to put it in/on dresser 1.
OK.
> go to dresser 1
On the dresser 1, you see a mug 1, and a television 1.
> put creditcard 2 in/on dresser 1
You put the creditcard 2 in/on the dresser 1.
> think: Now I put the first creditcard in dresser. Next, I need to find the second creditcard. I can directly go to countertop 1.
OK.
> go to countertop 1
On the countertop 1, you see a cellphone 2, a creditcard 4, a creditcard 3, a mirror 1, a pencil 2, and a pencil 1.
> think: Now I find the second creditcard (3). Next, I need to take it.
OK.
> take creditcard 3 from countertop 1
You pick up the creditcard 3 from the countertop 1.
> think: Now I take the second creditcard (3). Next, I need to put it in/on dresser 1.
OK.
> go to dresser 1
On the dresser 1, you see a creditcard 2, a mug 1, and a television 1.
> put creditcard 3 in/on dresser 1
You put the creditcard 3 in/on the dresser 1.
"""
    ),
    "look_at_obj_in_light": (
        """react_examine_0:
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a safe 1, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, a sidetable 2, and a sidetable 1.
Your task is to: look at bowl under the desklamp.
> think: To solve the task, I need to find and take a bowl, then find and use a desklamp. 
OK. 
> think: First I need to find a bowl. A bowl is more likely to appear in drawer (1-3), desk (1), sidetable (1-2), shelf (1-5), garbagecan (1). I can check one by one, starting with drawer 1.
OK.
> go to drawer 1
On the drawer 1, you see nothing.
> go to drawer 2
The drawer 2 is closed.
> open drawer 2
You open the drawer 2. The drawer 2 is open. In it, you see nothing.
> go to drawer 3
The drawer 3 is closed.
> open drawer 3
You open the drawer 3. The drawer 3 is open. In it, you see nothing.
> go to desk 1
On the desk 1, you see a alarmclock 2, a book 1, a cellphone 1, a keychain 1, a laptop 2, a laptop 1, and a pen 3.
> go to sidetable 1
On the sidetable 1, you see a cd 1, a pen 1, and a pencil 1.
> go to shelf 1
On the shelf 1, you see nothing.
> go to shelf 2
On the shelf 2, you see a bowl 1.
> think: Now I find a bowl (1). Next, I need to take it.
OK.
> take bowl 1 from shelf 2
You pick up the bowl 1 from the shelf 2.
> think: Now I take a bowl (1). Next, I need to find a desklamp. A desklamp is more likely to appear in desk (1), sidetable (1-2), shelf (1-5), bed (1), drawer (1-3). I can check one by one, starting with desk 1.
OK.
> go to desk 1
On the desk 1, you see a alarmclock 2, a book 1, a cellphone 1, a keychain 1, a laptop 2, a laptop 1, and a pen 3.
> go to sidetable 1
On the sidetable 1, you see a cd 1, a pen 1, and a pencil 1.
> go to sidetable 2
On the sidetable 2, you see a alarmclock 1, a desklamp 1, and a pen 2.
> think: Now I find a desklamp (1). Next, I need to use it.
OK.
> use desklamp 1
You turn on the desklamp 1.
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


class LlamaActionPolicy:
    def __init__(self, cfg: PolicyConfig) -> None:
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.max_new_tokens = cfg.max_new_tokens
        self.history_window = cfg.history_window
        self.prompting_mode = cfg.prompting_mode

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
    ) -> str:
        state_summary = self._summarize_trajectory(trajectory)

        goal_text = "unknown"
        goal_match = re.search(
            r"your task is to:?\s*(.+?)(?:[.\n]|$)",
            observation,
            flags=re.IGNORECASE,
        )
        if goal_match:
            goal_text = goal_match.group(1).strip()
        few_shot = REACT_FEW_SHOTS_BY_TASK.get(task_type, REACT_FEW_SHOTS_BY_TASK["pick_and_place_simple"])

        return (
            "You are an ALFWorld decision-making agent.\n"
            "Follow the few-shot style.\n"
            "Keep reasoning concise and output one next action.\n\n"

            f"Task type: {task_type}\n"
            f"{few_shot}\n"

            f"Current observation:\n{observation}\n\n"
            f"Your goal is: {goal_text}\n"

            f"State summary:\n{state_summary}\n\n"
            "Output format (must match):\n"
            "Thought: <1-2 concise sentences>\n"
            "Action: <one action>"
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

    @torch.no_grad()
    def _generate_text(self, messages: Sequence[Dict[str, str]]) -> str:
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
        gen_temperature = self.temperature if self.temperature > 0 else 0.7
        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=gen_temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
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
    ) -> Tuple[str, str]:

        prompt = self.build_prompt(
            observation,
            admissible_commands,
            trajectory,
            task_type,
        )

        messages = [
            {"role": "system", "content": "You are a precise ALFWorld agent."},
            {"role": "user", "content": prompt},
        ]
        raw_text = self._generate_text(messages)

        thought = self._extract_thought(raw_text)
        matched_action = self._match_action(raw_text, admissible_commands)

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
    )
    if reuse and cfg in _POLICY_CACHE:
        return _POLICY_CACHE[cfg]

    policy = LlamaActionPolicy(cfg)
    if reuse:
        _POLICY_CACHE[cfg] = policy
    return policy
