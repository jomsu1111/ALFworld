# 실행 예시:
# uv run batch.py --config configs/alfworld_llama3_zeroshot.yaml --split eval_out_of_distribution --seed 42 --num-games 100 --concurrency 4 --progress-every 10 --model meta-llama/Llama-3.1-8B-Instruct --hf-token <YOUR_HF_TOKEN>
# (기본값 사용 시) uv run batch.py

import argparse
import copy
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

from alfworld.agents.environment import get_environment
from llm_alfworld_utils import (
    build_turn_context,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from model_client import get_llama_action_policy


ENV_OP_LOCK = threading.Lock()
MODEL_OP_LOCK = threading.Lock()


def first_or_default(value, default=None):
    if isinstance(value, list):
        return value[0] if len(value) > 0 else default
    if value is None:
        return default
    return value


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_goal(turn_context):
    trajectory_context = turn_context.get("trajectory_context", {})
    return (
        turn_context.get("task_from_observation")
        or trajectory_context.get("human_task_desc")
        or trajectory_context.get("templated_task_desc")
        or "(목표 정보를 찾지 못했습니다)"
    )


def run_one_episode(env, policy, env_type, session_id):
    with ENV_OP_LOCK:
        obs, infos = env.reset()

    turn_idx = 0
    previous_action = "restart"
    last_printed_goal = None
    trajectory = []
    reflections = []

    while True:
        current_obs = obs[0]
        turn_context = build_turn_context(turn_idx, current_obs, infos, previous_action, env_type)
        admissible_commands = turn_context["admissible_commands"]
        task_type = turn_context.get("trajectory_context", {}).get("task_type", "") or "pick_and_place_simple"

        if not admissible_commands:
            raise RuntimeError("No admissible commands available in infos['admissible_commands'].")

        behavior_goal = resolve_goal(turn_context)
        if behavior_goal != last_printed_goal:
            print("[세션 {}] 행동 목표: {}".format(session_id, behavior_goal))
            last_printed_goal = behavior_goal

        with MODEL_OP_LOCK:
            command, _raw_model_output = policy.select_action(
                observation=current_obs,
                admissible_commands=admissible_commands,
                trajectory=trajectory,
                task_type=task_type,
                reflections=reflections,
            )
        if command not in admissible_commands:
            command = "look" if "look" in admissible_commands else random.choice(admissible_commands)

        print("[세션 {}] 파싱 응답: {}".format(session_id, command))

        with ENV_OP_LOCK:
            obs, scores, dones, infos = env.step([command])
        trajectory.append((command, obs[0]))

        if len(trajectory) >= 3 and trajectory[-1][0] == trajectory[-2][0] == trajectory[-3][0]:
            reflections.append("동일 액션 반복을 피하고 다른 admissible action으로 하위 목표를 전환하라.")
            reflections = reflections[-4:]

        previous_action = command
        turn_idx += 1

        if dones[0]:
            final_won = bool(first_or_default(infos.get("won"), False))
            final_gc_sr = first_or_default(infos.get("goal_condition_success_rate"), None)
            final_score = first_or_default(scores, None)
            return final_won, final_gc_sr, final_score


def run_single_game(
    game_file,
    game_index,
    session_id,
    env_prototype,
    env_type,
    seed,
    policy,
):
    print("\n[{} / ?] [세션 {}] Running game: {}".format(game_index, session_id, game_file))

    env_builder = copy.copy(env_prototype)
    env_builder.game_files = [game_file]
    env_builder.num_games = 1
    with ENV_OP_LOCK:
        env = env_builder.init_env(batch_size=1)

    if hasattr(env, "seed"):
        with ENV_OP_LOCK:
            env.seed(seed + game_index)

    try:
        won, gc_sr, score = run_one_episode(env, policy, env_type, session_id)
        return game_index, session_id, won, gc_sr, score
    finally:
        if hasattr(env, "close"):
            with ENV_OP_LOCK:
                env.close()


def main():
    parser = argparse.ArgumentParser(description="Run fixed-seed batch evaluation on ALFWorld eval split.")
    parser.add_argument(
        "--config",
        default="configs/alfworld_llama3_zeroshot.yaml",
        help="Path to ALFWorld config yaml",
    )
    parser.add_argument(
        "--split",
        default="eval_out_of_distribution",
        choices=["eval_in_distribution", "eval_out_of_distribution"],
        help="Eval split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to evaluate")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent sessions")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N games")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""), help="Hugging Face token")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError("Config file not found: {}".format(args.config))

    if args.progress_every <= 0:
        raise ValueError("--progress-every must be > 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")
    if not args.hf_token:
        raise ValueError("HF token is required. Pass --hf-token or set HF_TOKEN.")

    random.seed(args.seed)

    config = load_config(args.config)
    env_type = config["env"]["type"]
    print("Environment type: {}".format(env_type))
    print("Eval split: {}".format(args.split))
    print("Fixed seed: {}".format(args.seed))
    print("Concurrency: {}".format(args.concurrency))

    policy = get_llama_action_policy(
        model_id=args.model,
        hf_token=args.hf_token,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=64,
        history_window=8,
        prompting_mode="react",
        use_reflexion=True,
        reflection_window=4,
        reuse=True,
    )

    env_prototype = get_environment(env_type)(config, train_eval=args.split)
    all_games = list(getattr(env_prototype, "game_files", []))

    if len(all_games) == 0:
        raise RuntimeError("No games found in eval split: {}".format(args.split))

    if args.num_games > len(all_games):
        raise ValueError(
            "Requested {} games but only {} available in split {}".format(
                args.num_games,
                len(all_games),
                args.split,
            )
        )

    sampler = random.Random(args.seed)
    sampled_games = sampler.sample(all_games, args.num_games)

    print("Sampled {} games from eval split.".format(len(sampled_games)))

    successes = 0

    completed = 0
    for batch_start in range(0, args.num_games, args.concurrency):
        mini_batch = sampled_games[batch_start: batch_start + args.concurrency]

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = []
            for offset, game_file in enumerate(mini_batch):
                index = batch_start + offset + 1
                session_id = offset + 1
                futures.append(
                    executor.submit(
                        run_single_game,
                        game_file,
                        index,
                        session_id,
                        env_prototype,
                        env_type,
                        args.seed,
                        policy,
                    )
                )

            for future in as_completed(futures):
                index, session_id, won, gc_sr, score = future.result()
                completed += 1
                successes += 1 if won else 0

                print(
                    "[세션 {}] [{} / {}] 결과 - 성공: {}, goal_condition_success_rate: {}, score: {}".format(
                        session_id,
                        index,
                        args.num_games,
                        won,
                        gc_sr,
                        score,
                    )
                )

                if completed % args.progress_every == 0 or completed == args.num_games:
                    progress = (completed / args.num_games) * 100.0
                    current_success_rate = (successes / completed) * 100.0
                    print(
                        "진행률: {:.1f}% ({}/{}), 현재 성공률: {:.2f}% ({}/{})".format(
                            progress,
                            completed,
                            args.num_games,
                            current_success_rate,
                            successes,
                            completed,
                        )
                    )

    final_success_rate = (successes / args.num_games) * 100.0
    print("\n평가 완료")
    print("최종 성공률: {:.2f}% ({}/{})".format(final_success_rate, successes, args.num_games))


if __name__ == "__main__":
    main()
