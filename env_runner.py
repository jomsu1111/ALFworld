from typing import Dict, List, Tuple

from alfworld.agents.environment import get_environment

from alfworld_utils import map_split_name
from model_client import LlamaActionPolicy


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
        done = False
        step_count = 0
        trajectory: List[Tuple[str, str]] = []
        last_score = 0.0
        won = False

        while not done and step_count < max_steps:
            admissible = list(infos["admissible_commands"][0])
            action, raw_model_output = policy.select_action(observation, admissible, trajectory)
            obs, scores, dones, infos = env.step([action])
            next_observation = obs[0]
            done = bool(dones[0])
            last_score = float(scores[0])
            won = bool(infos.get("won", [False])[0]) or won
            trajectory.append((action, next_observation))
            observation = next_observation
            step_count += 1

            print(
                f"[episode {ep + 1:03d} step {step_count:02d}] action={action!r} "
                f"score={last_score:.3f} done={done} won={won} model={raw_model_output!r}"
            )

        success = bool(won or last_score >= 1.0)
        successes += int(success)
        total_score += last_score
        total_steps += step_count

        rec = {
            "episode": ep + 1,
            "success": success,
            "won": bool(won),
            "score": float(last_score),
            "steps": int(step_count),
        }
        results.append(rec)
        print(
            f"[episode {ep + 1:03d} done] success={success} score={last_score:.3f} steps={step_count}"
        )

    return {
        "results": results,
        "successes": successes,
        "total_score": total_score,
        "total_steps": total_steps,
    }
