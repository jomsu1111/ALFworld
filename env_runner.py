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
    if "closed" in observation.lower():
        return f"Action '{action}' failed due to closed receptacle. Prefer open-before-put/take when available."
    if "not carrying" in observation.lower():
        return f"Action '{action}' failed without inventory object. Prefer take before interact actions."
    if "nothing happens" in observation.lower() or "cannot" in observation.lower() or "can't" in observation.lower():
        return f"Action '{action}' caused no progress. Choose a different subgoal from admissible actions."
    if "look" in [cmd.lower() for cmd in admissible]:
        return f"Action '{action}' did not improve state. Use look/explore/open to reveal missing affordances."
    return f"Action '{action}' did not improve progress. Avoid repeating it in same state."


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
    low_value = {"inventory", "look", "examine"}
    norm_action = action.strip().lower()
    if norm_action not in low_value:
        return action

    if repeated_action_streak < 2 and no_progress_steps < 2:
        return action

    preferred = [
        cmd for cmd in admissible
        if cmd.strip().lower() not in low_value and cmd.strip().lower() != norm_action
    ]
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

        while not done and step_count < max_steps:
            admissible = list(infos["admissible_commands"][0])
            action, _raw_model_output = policy.select_action(
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
            trajectory.append((action, next_observation))

            if action == prev_action:
                repeated_action_streak += 1
            else:
                repeated_action_streak = 1
            prev_action = action

            if next_observation.strip() == prev_observation.strip():
                stagnant_observation_streak += 1
            else:
                stagnant_observation_streak = 0

            progressed = bool(last_score > prev_score or won)
            if progressed:
                no_progress_steps = 0
            else:
                no_progress_steps += 1

            if repeated_action_streak >= 3 or stagnant_observation_streak >= 3:
                reflection = (
                    "Immediate reflection: loop detected (same action or unchanged observation >= 3). "
                    "Switch subgoal and avoid repeated look/examine."
                )
                _append_unique_reflection(reflections, reflection)
                repeated_action_streak = 0
                stagnant_observation_streak = 0

            if no_progress_steps >= 5:
                reflection = (
                    f"Mid-episode reflection: no progress for {no_progress_steps} steps. "
                    "Mark completed subgoals explicitly, pick next subgoal, then execute a different admissible action."
                )
                _append_unique_reflection(reflections, reflection)
                no_progress_steps = 0

            if last_score <= prev_score and _is_stuck_transition(action, next_observation, prev_observation):
                _append_unique_reflection(reflections, _build_reflection(action, next_observation, admissible))

            observation = next_observation
            step_count += 1

            print(
                f"[episode {ep + 1:03d} step {step_count:02d}] task={task_type} action={action!r} "
                f"score={last_score:.3f} done={done} won={won}"
            )

        success = bool(won or last_score >= 1.0)
        successes += int(success)
        total_score += last_score
        total_steps += step_count

        rec = {
            "episode": ep + 1,
            "task_type": task_type,
            "success": success,
            "won": bool(won),
            "score": float(last_score),
            "steps": int(step_count),
            "reflections": reflections,
        }
        results.append(rec)
        print(
            f"[episode {ep + 1:03d} done] task={task_type} success={success} "
            f"score={last_score:.3f} steps={step_count}"
        )

    return {
        "results": results,
        "successes": successes,
        "total_score": total_score,
        "total_steps": total_steps,
    }
