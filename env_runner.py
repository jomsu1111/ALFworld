import re
from typing import Dict, List, Set, Tuple

from alfworld.agents.environment import get_environment

from alfworld_utils import map_split_name
from model_client import LlamaActionPolicy


def _infer_task_type(infos: Dict, observation: str) -> str:
    gamefile_candidates = []
    if "extra.gamefile" in infos and infos["extra.gamefile"]:
        gamefile_candidates.extend([str(x) for x in infos["extra.gamefile"] if x])
    if "gamefile" in infos and infos["gamefile"]:
        gamefile_candidates.extend([str(x) for x in infos["gamefile"] if x])

    joined = " ".join(gamefile_candidates).lower() + " " + observation.lower()
    task_types = [
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
    ]
    for task in task_types:
        if task in joined:
            return task

    if re.search(r"\bclean\b", joined):
        return "pick_clean_then_place_in_recep"
    if re.search(r"\bheat|hot\b", joined):
        return "pick_heat_then_place_in_recep"
    if re.search(r"\bcool|cold|chill\b", joined):
        return "pick_cool_then_place_in_recep"
    if re.search(r"\blight|lamp\b", joined):
        return "look_at_obj_in_light"
    if re.search(r"\btwo\b", joined):
        return "pick_two_obj_and_place"
    return "pick_and_place_simple"


def _is_stuck_transition(action: str, next_observation: str, prev_observation: str) -> bool:
    text = next_observation.lower()
    stuck_signals = (
        "nothing happens",
        "you can't",
        "cannot",
        "not possible",
        "can't see any",
        "there is no",
        "you are not carrying",
        "is closed",
    )
    if any(sig in text for sig in stuck_signals):
        return True
    return next_observation.strip() == prev_observation.strip()


def _build_reflection(action: str, observation: str, admissible: List[str]) -> str:
    obs = observation.lower()
    action = action.lower()

    if "closed" in obs and ("put" in action or "take" in action):
        return (
            f"Action '{action}' failed because the receptacle was closed. "
            "Open it before interacting."
        )
    if "not carrying" in obs or "not holding" in obs:
        return (
            f"Action '{action}' failed because no object was held. "
            "Take the required object first."
        )
    if "nothing happens" in obs:
        return (
            f"Action '{action}' caused no state change. "
            "You may be at the wrong location or missing a precondition."
        )
    if "cannot" in obs or "can't" in obs:
        return (
            f"Action '{action}' is invalid in this state. "
            "Check location, object possession, or receptacle state."
        )
    return (
        f"Action '{action}' did not help progress. "
        "Avoid repeating it in the same state."
    )


def _append_unique_reflection(reflections: List[str], text: str, max_len: int = 8) -> None:
    if not text:
        return
    if not reflections or reflections[-1] != text:
        reflections.append(text)
    if len(reflections) > max_len:
        del reflections[:-max_len]


def _normalize_goal_object_phrase(text: str) -> str:
    if not text:
        return text
    return re.sub(
        r"^(?:a|an|the|one|two|three|four|five|\d+)\s+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

def _parse_goal_semantics(task_type: str, goal_object: str):
    """
    Splits goal_object into:
    - base_object (e.g., "plate")
    - required_state (e.g., "cool", "heat", "clean", or None)
    """
    if not goal_object:
        return None, None

    goal_object = goal_object.lower().strip()

    required_state = None
    base_object = goal_object

    if task_type == "pick_cool_then_place_in_recep":
        required_state = "cool"
    elif task_type == "pick_heat_then_place_in_recep":
        required_state = "heat"
    elif task_type == "pick_clean_then_place_in_recep":
        required_state = "clean"

    # remove state adjective if present
    if required_state and goal_object.startswith(required_state + " "):
        base_object = goal_object[len(required_state) + 1 :]

    return base_object.strip(), required_state

def _build_stage_plan(task_type: str, goal_object: str, goal_receptacle: str) -> List[str]:
    base_object, required_state = _parse_goal_semantics(task_type, goal_object)

    go = base_object or "unknown_goal_object"
    gr = goal_receptacle or "unknown_goal_receptacle"

    # ---- SIMPLE PICK & PLACE ----
    if task_type == "pick_and_place_simple":
        return [
            f"Stage 1: Locate '{go}'",
            f"Stage 2: Take '{go}'",
            f"Stage 3: Go to receptacle '{gr}'",
            f"Stage 4: Open receptacle '{gr}' if needed",
            f"Stage 5: Place '{go}' into '{gr}'",
        ]

    # ---- TRANSFORMATION TASKS (clean / heat / cool) ----
    if task_type in [
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
    ]:
        appliance_stage = {
            "clean": "Go to cleaning appliance",
            "heat": "Go to heating appliance",
            "cool": "Go to cooling appliance",
        }

        transform_stage = {
            "clean": f"Clean '{go}'",
            "heat": f"Heat '{go}'",
            "cool": f"Cool '{go}'",
        }

        return [
            f"Stage 1: Locate '{go}'",
            f"Stage 2: Take '{go}'",
            f"Stage 3: {appliance_stage[required_state]}",
            f"Stage 4: {transform_stage[required_state]}",
            f"Stage 5: Go to receptacle '{gr}'",
            f"Stage 6: Open receptacle '{gr}' if needed",
            f"Stage 7: Place '{go}' into '{gr}'",
        ]

    # ---- LOOK TASK ----
    if task_type == "look_at_obj_in_light":
        return [
            f"Stage 1: Locate '{go}'",
            f"Stage 2: Take '{go}'",
            f"Stage 3: Go to light source",
            f"Stage 4: Turn light on",
            f"Stage 5: Examine '{go}' under light",
        ]

    # ---- TWO OBJECTS ----
    if task_type == "pick_two_obj_and_place":
        return [
            f"Stage 1: Locate '{go}'",
            f"Stage 2: Take '{go}'",
            f"Stage 3: Go to receptacle '{gr}'",
            f"Stage 4: Open receptacle '{gr}' if needed",
            f"Stage 5: Place '{go}' into '{gr}'",
            f"Stage 6: Repeat until required count satisfied",
        ]

    # fallback
    return [
        f"Stage 1: Locate '{go}'",
        f"Stage 2: Take '{go}'",
        f"Stage 3: Go to receptacle '{gr}'",
        f"Stage 4: Open receptacle '{gr}' if needed",
        f"Stage 5: Place '{go}' into '{gr}'",
    ]

def _required_object_count(task_type: str, goal_text: str) -> int:
    if task_type == "pick_two_obj_and_place":
        return 2
    if goal_text and re.search(r"\btwo\b|\b2\b", goal_text.lower()):
        return 2
    return 1


def _find_stage_index(stage_plan: List[str], keyword: str, default: int) -> int:
    for i, s in enumerate(stage_plan):
        if keyword in s.lower():
            return i
    return default


def _sync_stage_from_state(
    *,
    stage_plan: List[str],
    goal_visible: bool,
    holding_goal_object: bool,
    at_receptacle: bool,
    can_open_target: bool,
    can_put_target: bool,
) -> int:
    idx_locate = _find_stage_index(stage_plan, "locate object instances", 0)
    idx_take = _find_stage_index(stage_plan, "take object", min(1, len(stage_plan) - 1))
    idx_goto = _find_stage_index(stage_plan, "go to target receptacle", min(2, len(stage_plan) - 1))
    idx_open = _find_stage_index(stage_plan, "open target receptacle", idx_goto)
    idx_place = _find_stage_index(stage_plan, "place object", len(stage_plan) - 1)

    if holding_goal_object:
        if at_receptacle:
            if can_put_target:
                return idx_place
            if can_open_target:
                return idx_open
            return idx_goto
        return idx_goto
    if goal_visible:
        return idx_take
    return idx_locate


def _stage_completed(
    stage_text: str,
    *,
    action: str,
    next_observation: str,
    next_admissible: List[str],
    holding_object: str,
    goal_object_key: str,
    goal_receptacle_key: str,
    prev_score: float,
    last_score: float,
    won: bool,
) -> bool:
    stage_l = stage_text.lower()
    action_l = action.lower()
    obs_l = next_observation.lower()
    goal_object_pattern = rf"\b{re.escape(goal_object_key)}\b" if goal_object_key else ""

    if "locate" in stage_l:
        if goal_object_pattern and re.search(goal_object_pattern, obs_l):
            return True
        if goal_object_key:
            return any(
                cmd.lower().startswith("take ") and re.search(goal_object_pattern, cmd.lower())
                for cmd in next_admissible
            )
        return "take " in "\n".join(next_admissible).lower()
    if "take goal object" in stage_l or "take object" in stage_l:
        return bool(
            holding_object
            and (not goal_object_pattern or re.search(goal_object_pattern, holding_object.lower()))
        )
    if "go to target receptacle" in stage_l:
        if goal_receptacle_key and action_l.startswith("go to") and goal_receptacle_key in action_l:
            return True
        return bool(goal_receptacle_key and goal_receptacle_key in obs_l)
    if "go to light source" in stage_l:
        return action_l.startswith("go to") and ("light" in action_l or "lamp" in action_l)
    if "go to cleaning appliance" in stage_l:
        return action_l.startswith("go to") and ("sink" in action_l or "basin" in action_l)
    if "go to heating appliance" in stage_l:
        return action_l.startswith("go to") and ("microwave" in action_l or "stove" in action_l or "oven" in action_l)
    if "go to cooling appliance" in stage_l:
        return action_l.startswith("go to") and ("fridge" in action_l or "refrigerator" in action_l)
    if "open target receptacle" in stage_l:
        if goal_receptacle_key and action_l.startswith("open ") and goal_receptacle_key in action_l:
            return True
        if goal_receptacle_key:
            return any(cmd.lower().startswith("put ") and goal_receptacle_key in cmd.lower() for cmd in next_admissible)
        return any(cmd.lower().startswith("put ") for cmd in next_admissible)
    if "clean goal object" in stage_l:
        return action_l.startswith("clean ") and "nothing happens" not in obs_l
    if "heat goal object" in stage_l:
        return action_l.startswith("heat ") and "nothing happens" not in obs_l
    if "cool goal object" in stage_l:
        return action_l.startswith("cool ") and "nothing happens" not in obs_l
    if "toggle light on" in stage_l:
        return action_l.startswith("toggle ") and ("light" in action_l or "lamp" in action_l)
    if "examine target object" in stage_l:
        return action_l.startswith("examine ") and "nothing happens" not in obs_l
    if "place goal object" in stage_l or "place object" in stage_l:
        return action_l.startswith(("put ", "move ")) and (
            "nothing happens" not in obs_l or last_score > prev_score or won
        )
    if "repeat until count satisfied" in stage_l:
        return False
    return False


def _filter_admissible_for_subgoal(
    admissible: List[str],
    subgoal: str,
    *,
    goal_object_key: str,
    goal_receptacle_key: str,
    holding_object: str,
    checked_locations: Set[str],
) -> List[str]:
    subgoal_l = subgoal.lower()
    goal_object_pattern = rf"\b{re.escape(goal_object_key)}\b" if goal_object_key else ""

    def keep(cmd: str) -> bool:
        c = cmd.lower().strip()
        if "locate" in subgoal_l:
            if c.startswith("take "):
                return bool(goal_object_pattern and re.search(goal_object_pattern, c))
            if c.startswith("examine "):
                return c[len("examine ") :].strip() not in checked_locations
            if c.startswith("open "):
                return c[len("open ") :].strip() not in checked_locations
            return c.startswith("look") or c.startswith("go to") or c.startswith("examine") or c.startswith("open ")
        if "take goal object" in subgoal_l or "take object" in subgoal_l:
            if goal_object_key:
                return c.startswith("take ") and bool(re.search(goal_object_pattern, c))
            return c.startswith("take ")
        if "go to target receptacle" in subgoal_l:
            return c.startswith("go to") and (not goal_receptacle_key or goal_receptacle_key in c)
        if "open target receptacle" in subgoal_l:
            return c.startswith("open ") and (not goal_receptacle_key or goal_receptacle_key in c)
        if "place goal object" in subgoal_l or "place object" in subgoal_l:
            return c.startswith(("put ", "move ")) and (not goal_receptacle_key or goal_receptacle_key in c)
        if "go to light source" in subgoal_l:
            return c.startswith("go to") and ("light" in c or "lamp" in c)
        if "toggle light on" in subgoal_l:
            return c.startswith("toggle ") and ("light" in c or "lamp" in c)
        if "examine target object" in subgoal_l:
            return c.startswith("examine ")
        if "go to cleaning appliance" in subgoal_l:
            return c.startswith("go to") and ("sink" in c or "basin" in c)
        if "clean goal object" in subgoal_l:
            return c.startswith("clean ")
        if "go to heating appliance" in subgoal_l:
            return c.startswith("go to") and ("microwave" in c or "stove" in c or "oven" in c)
        if "heat goal object" in subgoal_l:
            return c.startswith("heat ")
        if "go to cooling appliance" in subgoal_l:
            return c.startswith("go to") and ("fridge" in c or "refrigerator" in c)
        if "cool goal object" in subgoal_l:
            return c.startswith("cool ")
        if "repeat until count satisfied" in subgoal_l:
            return c.startswith("take ") or c.startswith("go to") or c.startswith("open ") or c.startswith("put ")
        return True

    filtered = [cmd for cmd in admissible if keep(cmd)]
    return filtered if filtered else admissible


def _rerank_loop_prone_action(
    action: str,
    admissible: List[str],
    repeated_action_streak: int,
    no_progress_steps: int,
) -> str:
    norm_action = action.strip().lower()
    is_low_value = any(keyword in norm_action for keyword in ["inventory", "look", "examine"])
    if not is_low_value:
        return action
    if repeated_action_streak < 2 and no_progress_steps < 2:
        return action
    priority_keywords = ["take", "put", "open", "go to", "heat", "cool", "clean"]
    preferred = []
    for cmd in admissible:
        cmd_lower = cmd.strip().lower()
        if cmd_lower == norm_action:
            continue
        if any(k in cmd_lower for k in priority_keywords):
            preferred.append(cmd)
    if preferred:
        return preferred[0]
    alternatives = [cmd for cmd in admissible if cmd.strip().lower() != norm_action]
    if alternatives:
        return alternatives[0]
    return action


def build_env(config: Dict, split: str):
    split_name = map_split_name(split)
    env = get_environment(config["env"]["type"])(config, train_eval=split_name)
    return env.init_env(batch_size=1)


def run_episodes(
    *,
    env,
    policy: LlamaActionPolicy,
    episodes: int,
    max_steps: int,
) -> Dict:
    results = []
    successes = 0
    total_score = 0.0
    total_steps = 0

    for ep in range(episodes):
        obs, infos = env.reset()
        observation = obs[0]
        initial_goal_match = re.search(
            r"your task is to:?\s*(.+?)(?:[.\n]|$)",
            observation,
            flags=re.IGNORECASE,
        )
        episode_goal_text = initial_goal_match.group(1).strip() if initial_goal_match else None
        task_type = _infer_task_type(infos, observation)
        done = False
        step_count = 0
        trajectory: List[Tuple[str, str]] = []
        reflections: List[str] = []

        last_score = 0.0
        won = False

        no_progress_steps = 0
        repeated_action_streak = 0
        stagnant_observation_streak = 0
        prev_action = ""
        goal_match = re.search(
            r"your task is to:?\s*put\s+(.+?)\s+(?:in|on)\s+(.+?)(?:[.\n]|$)",
            observation,
            flags=re.IGNORECASE,
        )
        goal_object = None
        goal_receptacle = None
        if goal_match:
            goal_object = _normalize_goal_object_phrase(
                re.sub(r"^(a|an|the)\s+", "", goal_match.group(1).strip(), flags=re.IGNORECASE)
            )
            goal_receptacle = goal_match.group(2).strip()
        goal_object_key = (
            _normalize_goal_object_phrase(re.sub(r"^(a|an|the)\s+", "", goal_object, flags=re.IGNORECASE)).lower()
            if goal_object
            else None
        )
        goal_receptacle_key = (
            re.sub(r"^(a|an|the)\s+", "", goal_receptacle, flags=re.IGNORECASE).lower()
            if goal_receptacle
            else None
        )

        holding_object = None
        stage_plan = _build_stage_plan(task_type, goal_object, goal_receptacle)
        stage_idx = 0
        current_subgoal = stage_plan[stage_idx]
        stage_feedback = ""
        required_count = _required_object_count(task_type, episode_goal_text or "")
        placed_count = 0
        idx_take = _find_stage_index(stage_plan, "take object", 1)
        checked_locations: Set[str] = set()

        print(
            f"[episode {ep + 1:03d} start] "
            f"task={episode_goal_text or 'unknown'} task_type={task_type}"
        )

        while not done and step_count < max_steps:
            admissible = list(infos["admissible_commands"][0])
            obs_l = observation.lower()
            goal_object_pattern = rf"\b{re.escape(goal_object_key)}\b" if goal_object_key else ""
            goal_visible = bool(
                goal_object_key
                and (
                    re.search(goal_object_pattern, obs_l)
                    or any(
                        cmd.lower().startswith("take ") and re.search(goal_object_pattern, cmd.lower())
                        for cmd in admissible
                    )
                )
            )
            holding_goal_object = bool(
                holding_object
                and (not goal_object_pattern or re.search(goal_object_pattern, holding_object.lower()))
            )
            at_receptacle = bool(goal_receptacle_key and goal_receptacle_key in obs_l)
            can_open_target = any(
                cmd.lower().startswith("open ")
                and (not goal_receptacle_key or goal_receptacle_key in cmd.lower())
                for cmd in admissible
            )
            can_put_target = any(
                cmd.lower().startswith("put ")
                and (not goal_receptacle_key or goal_receptacle_key in cmd.lower())
                for cmd in admissible
            )

            stage_idx = _sync_stage_from_state(
                stage_plan=stage_plan,
                goal_visible=goal_visible,
                holding_goal_object=holding_goal_object,
                at_receptacle=at_receptacle,
                can_open_target=can_open_target,
                can_put_target=can_put_target,
            )
            current_subgoal = stage_plan[min(stage_idx, len(stage_plan) - 1)]

            if holding_object is not None:
                blocked_keywords = ("look", "examine", "inventory")
                admissible = [
                    cmd for cmd in admissible
                    if not any(keyword in cmd.lower() for keyword in blocked_keywords)
                ]
            admissible = _filter_admissible_for_subgoal(
                admissible,
                current_subgoal,
                goal_object_key=goal_object_key,
                goal_receptacle_key=goal_receptacle_key,
                holding_object=holding_object,
                checked_locations=checked_locations,
            )
            is_locate_stage = "locate" in current_subgoal.lower()

            inferred_subgoal = getattr(policy, "last_inferred_subgoal", "")
            goal_prefix = f"Episode goal: {episode_goal_text or 'unknown'}\n"
            inferred_subgoal_line = (
                f"Model inferred subgoal: {inferred_subgoal}\n"
                if inferred_subgoal
                else ""
            )
            stage_feedback_line = f"{stage_feedback}\n" if stage_feedback else ""
            observation_for_policy = (
                f"{goal_prefix}"
                f"{stage_feedback_line}"
                f"Current subgoal: {current_subgoal}\n"
                f"{inferred_subgoal_line}"
                f"Progress: placed_count={placed_count}/{required_count}\n"
                f"{observation}"
            )

            action, thought = policy.select_action(
                observation=observation_for_policy,
                admissible_commands=admissible,
                trajectory=trajectory,
                task_type=task_type,
                reflections=reflections,
            )

            action = _rerank_loop_prone_action(
                action=action,
                admissible=admissible,
                repeated_action_streak=repeated_action_streak,
                no_progress_steps=no_progress_steps,
            )

            prev_observation = observation
            prev_score = last_score

            obs, scores, dones, infos = env.step([action])
            next_observation = obs[0]
            done = bool(dones[0])
            last_score = float(scores[0])
            won = bool(infos.get("won", [False])[0]) or won

            if next_observation == prev_observation and last_score <= prev_score and not done:
                admissible = [
                    cmd for cmd in admissible
                    if cmd.strip().lower() != action.strip().lower()
                ]
                if admissible:
                    action, thought = policy.select_action(
                        observation=observation_for_policy,
                        admissible_commands=admissible,
                        trajectory=trajectory,
                        task_type=task_type,
                        reflections=reflections,
                    )
                    obs, scores, dones, infos = env.step([action])
                    next_observation = obs[0]
                    done = bool(dones[0])
                    last_score = float(scores[0])
                    won = bool(infos.get("won", [False])[0]) or won

            trajectory.append((action, next_observation))
            if action.startswith("take "):
                taken = action[len("take "):].split(" from ", 1)[0].strip()
                holding_object = goal_object if goal_object else taken
            elif action.startswith(("put ", "move ")):
                holding_object = None

            next_admissible = list(infos.get("admissible_commands", [[""]])[0])
            if is_locate_stage and goal_object_key and action.lower().startswith(("examine ", "open ")):
                visited_location = action.split(" ", 1)[1].strip().lower()
                found_goal_here = bool(re.search(goal_object_pattern, next_observation.lower()))
                if not found_goal_here:
                    checked_locations.add(visited_location)
            if stage_idx < len(stage_plan):
                current_stage = stage_plan[stage_idx]
                if _stage_completed(
                    current_stage,
                    action=action,
                    next_observation=next_observation,
                    next_admissible=next_admissible,
                    holding_object=holding_object,
                    goal_object_key=goal_object_key,
                    goal_receptacle_key=goal_receptacle_key,
                    prev_score=prev_score,
                    last_score=last_score,
                    won=won,
                ):
                    stage_feedback = f"Great job. Completed stage: {current_stage}"
                    print(f"  praise={stage_feedback}")
                    if "place object" in current_stage.lower() or "place goal object" in current_stage.lower():
                        placed_count += 1
                    stage_idx += 1

            if stage_idx < len(stage_plan) and "repeat until count satisfied" in stage_plan[stage_idx].lower():
                if placed_count >= required_count:
                    stage_idx += 1
                else:
                    stage_idx = idx_take

            if stage_idx < len(stage_plan):
                current_subgoal = stage_plan[stage_idx]
            else:
                current_subgoal = "Task complete"

            repeated_action_streak = repeated_action_streak + 1 if action == prev_action else 1
            prev_action = action

            stagnant_observation_streak = (
                stagnant_observation_streak + 1
                if next_observation.strip() == prev_observation.strip()
                else 0
            )

            progressed = (
                last_score > prev_score
                or won
                or next_observation != prev_observation
            )
            if progressed:
                no_progress_steps = 0
            else:
                no_progress_steps += 1

            if repeated_action_streak >= 3 or stagnant_observation_streak >= 3:
                reflection = "Loop detected. Avoid repeating the same action."
                _append_unique_reflection(reflections, reflection)
                repeated_action_streak = 0
                stagnant_observation_streak = 0
            elif no_progress_steps >= 5:
                reflection = "No progress for several steps. Switch to a different admissible action."
                _append_unique_reflection(reflections, reflection)
                no_progress_steps = 0
            elif last_score <= prev_score and _is_stuck_transition(action, next_observation, prev_observation):
                _append_unique_reflection(
                    reflections,
                    _build_reflection(action, next_observation, admissible),
                )

            observation = next_observation
            step_count += 1

            print(
                f"[episode {ep + 1:03d} step {step_count:02d}] "
                f"action={action!r} score={last_score:.3f} "
                f"done={done} won={won}"
            )
            print(f"  subgoal={current_subgoal}")
            print(f"  thought={thought}")

        success = bool(won or last_score >= 1.0)
        successes += int(success)
        total_score += last_score
        total_steps += step_count

        results.append(
            {
                "episode": ep + 1,
                "success": success,
                "won": bool(won),
                "score": float(last_score),
                "steps": int(step_count),
                "reflections": reflections,
            }
        )

        print(
            f"[episode {ep + 1:03d} done] "
            f"success={success} score={last_score:.3f} steps={step_count}"
        )

    return {
        "results": results,
        "successes": successes,
        "total_score": total_score,
        "total_steps": total_steps,
    }
