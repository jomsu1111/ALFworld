import re
from typing import Dict, List, Protocol, Tuple

from alfworld.agents.environment import get_environment

from alfworld_utils import map_split_name


class ActionPolicy(Protocol):
    def select_action(
        self,
        observation: str,
        admissible_commands,
        trajectory,
        task_type: str = "pick_and_place_simple",
        goal_text: str | None = None,
    ) -> Tuple[str, str]:
        ...


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


def build_env(config: Dict, split: str):
    split_name = map_split_name(split)
    env = get_environment(config["env"]["type"])(config, train_eval=split_name)
    return env.init_env(batch_size=1)


def run_episodes(
    *,
    env,
    policy: ActionPolicy,
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

        goal_match = re.search(
            r"your task is to:?\s*(.+?)(?:[.\n]|$)",
            observation,
            flags=re.IGNORECASE,
        )
        episode_goal_text = goal_match.group(1).strip() if goal_match else "unknown"

        done = False
        step_count = 0
        trajectory: List[Tuple[str, str]] = []
        last_score = 0.0
        won = False

        print(f"[episode {ep + 1:03d} start] task={episode_goal_text} task_type={task_type}")

        while not done and step_count < max_steps:
            admissible = list(infos["admissible_commands"][0])

            action, thought = policy.select_action(
                observation=observation,
                admissible_commands=admissible,
                trajectory=trajectory,
                task_type=task_type,
                goal_text=episode_goal_text,
            )

            obs, scores, dones, infos = env.step([action])
            next_observation = obs[0]
            done = bool(dones[0])
            last_score = float(scores[0])
            won = bool(infos.get("won", [False])[0]) or won

            trajectory.append((action, next_observation))
            observation = next_observation
            step_count += 1

            print(
                f"[episode {ep + 1:03d} step {step_count:02d}] "
                f"action={action!r} score={last_score:.3f} done={done} won={won}"
            )
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
