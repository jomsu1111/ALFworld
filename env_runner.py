import re
from typing import Dict, List, Tuple

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

    # Fallback heuristic from instruction wording.
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

    # 1️⃣ Closed receptacle interaction
    if "closed" in obs and ("put" in action or "take" in action):
        return (
            f"Action '{action}' failed because the receptacle was closed. "
            "Open it before interacting."
        )

    # 2️⃣ Inventory failure
    if "not carrying" in obs or "not holding" in obs:
        return (
            f"Action '{action}' failed because no object was held. "
            "Take the required object first."
        )

    # 3️⃣ Nothing happens → likely wrong location
    if "nothing happens" in obs:
        return (
            f"Action '{action}' caused no state change. "
            "You may be at the wrong location or missing a precondition."
        )

    # 4️⃣ Cannot / can't
    if "cannot" in obs or "can't" in obs:
        return (
            f"Action '{action}' is invalid in this state. "
            "Check location, object possession, or receptacle state."
        )

    # 5️⃣ Default: avoid repetition
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


def _rerank_loop_prone_action(
    action: str,
    admissible: List[str],
    repeated_action_streak: int,
    no_progress_steps: int,
) -> str:

    norm_action = action.strip().lower()

    # 🔥 substring 기반 low-value 판별
    is_low_value = any(
        keyword in norm_action
        for keyword in ["inventory", "look", "examine"]
    )

    if not is_low_value:
        return action

    # 🔥 loop 강도 높이면 더 빠르게 차단
    if repeated_action_streak < 2 and no_progress_steps < 2:
        return action

    # 🔥 put/take/open/go to 우선
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

    # fallback: 다른 아무 action
    alternatives = [
        cmd for cmd in admissible
        if cmd.strip().lower() != norm_action
    ]

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
            r"your task is to:\s*put\s+(.+?)\s+in\s+(.+?)(?:[.\n]|$)",
            observation,
            flags=re.IGNORECASE,
        )
        goal_object = None
        goal_receptacle = None
        if goal_match:
            goal_object = re.sub(
                r"^(a|an|the)\s+",
                "",
                goal_match.group(1).strip(),
                flags=re.IGNORECASE,
            )
            goal_receptacle = goal_match.group(2).strip()

        holding_object = None
        stage = "find"

        while not done and step_count < max_steps:
            admissible = list(infos["admissible_commands"][0])
            if holding_object is not None:
                blocked_keywords = ("look", "examine", "inventory")
                admissible = [
                    cmd for cmd in admissible
                    if not any(keyword in cmd.lower() for keyword in blocked_keywords)
                ]

            # Stage constraint
            if stage == "goto_target":
                constrained = [
                    cmd for cmd in admissible
                    if (
                        (goal_receptacle and goal_receptacle.lower() in cmd.lower())
                        or cmd.lower().startswith("go to")
                        or cmd.lower().startswith("open")
                    )
                ]
                if constrained:
                    admissible = constrained
            elif stage == "place":
                constrained = [
                    cmd for cmd in admissible
                    if cmd.lower().startswith("put") or cmd.lower().startswith("open")
                ]
                if constrained:
                    admissible = constrained

            action, _ = policy.select_action(
                observation=observation,
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

            # Regeneration logic
            if (
                next_observation == prev_observation
                and last_score <= prev_score
                and not done
            ):
                admissible = [
                    cmd for cmd in admissible
                    if cmd.strip().lower() != action.strip().lower()
                ]
                if admissible:
                    action, _ = policy.select_action(
                        observation=observation,
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
                holding_object = goal_object
                stage = "goto_target"
            elif action.startswith("put "):
                holding_object = None
                stage = "done"
            if (
                holding_object is not None
                and action.lower().startswith("go to")
                and goal_receptacle
                and goal_receptacle.lower() in action.lower()
            ):
                stage = "place"

            # ===== Loop tracking =====
            repeated_action_streak = (
                repeated_action_streak + 1 if action == prev_action else 1
            )
            prev_action = action

            stagnant_observation_streak = (
                stagnant_observation_streak + 1
                if next_observation.strip() == prev_observation.strip()
                else 0
            )

            # ===== Progress 판단 완화 =====
            progressed = (
                last_score > prev_score
                or won
                or next_observation != prev_observation
            )

            if progressed:
                no_progress_steps = 0
            else:
                no_progress_steps += 1

            # ===== Minimal Reflexion =====
            if repeated_action_streak >= 3 or stagnant_observation_streak >= 3:
                reflection = (
                    "Loop detected. Avoid repeating the same action."
                )
                _append_unique_reflection(reflections, reflection)
                repeated_action_streak = 0
                stagnant_observation_streak = 0

            elif no_progress_steps >= 5:
                reflection = (
                    "No progress for several steps. Switch to a different admissible action."
                )
                _append_unique_reflection(reflections, reflection)
                no_progress_steps = 0

            # ===== Stuck transition reflection (간단화) =====
            elif last_score <= prev_score and _is_stuck_transition(
                action, next_observation, prev_observation
            ):
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
