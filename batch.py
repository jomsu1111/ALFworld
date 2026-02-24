# 실행 예시:
# uv run batch.py --config alfworld/configs/eval_config.yaml --split eval_out_of_distribution --seed 42 --num-games 100 --concurrency 4 --progress-every 10 --base-url http://localhost:8000/v1/ --api-key <YOUR_API_KEY> --model meta-llama/Llama-3.1-8B-Instruct
# (기본값 사용 시) uv run batch.py

import argparse
import copy
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from openai import OpenAI

from alfworld.agents.environment import get_environment
from llm_alfworld_utils import (
    build_turn_context,
    normalize_model_output_to_command,
    strip_think_tags,
    trim_history,
)


ENV_OP_LOCK = threading.Lock()


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


def run_one_episode(env, client, model_name, env_type, session_id):
    with ENV_OP_LOCK:
        obs, infos = env.reset()

    system_prompt = (
        "You are an agent that interacts with an environment. "
        "The environment provides observations, task/goal context, and a list of admissible commands. "
        "Always pick exactly one command from the admissible commands list. "
        "Return only the command text and nothing else."
    )

    messages = [{"role": "system", "content": system_prompt}]
    turn_idx = 0
    previous_action = "restart"
    last_printed_goal = None

    while True:
        current_obs = obs[0]
        turn_context = build_turn_context(turn_idx, current_obs, infos, previous_action, env_type)
        admissible_commands = turn_context["admissible_commands"]

        if not admissible_commands:
            raise RuntimeError("No admissible commands available in infos['admissible_commands'].")

        behavior_goal = resolve_goal(turn_context)
        if behavior_goal != last_printed_goal:
            print("[세션 {}] 행동 목표: {}".format(session_id, behavior_goal))
            last_printed_goal = behavior_goal

        user_payload = {
            "instruction": "Choose exactly one command from admissible_commands.",
            "context": turn_context,
        }
        messages.append({
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, indent=2)
        })
        messages = trim_history(messages)

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=4000,
        )

        raw_model_output = response.choices[0].message.content or ""
        model_output = strip_think_tags(raw_model_output)
        command = normalize_model_output_to_command(model_output, admissible_commands)
        if not command:
            command = "look" if "look" in admissible_commands else random.choice(admissible_commands)

        print("[세션 {}] 파싱 응답: {}".format(session_id, command))

        messages.append({"role": "assistant", "content": command})

        with ENV_OP_LOCK:
            obs, scores, dones, infos = env.step([command])
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
    base_url,
    api_key,
    model_name,
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

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    try:
        won, gc_sr, score = run_one_episode(env, client, model_name, env_type, session_id)
        return game_index, session_id, won, gc_sr, score
    finally:
        if hasattr(env, "close"):
            with ENV_OP_LOCK:
                env.close()


def main():
    parser = argparse.ArgumentParser(description="Run fixed-seed batch evaluation on ALFWorld eval split.")
    parser.add_argument(
        "--config",
        default="alfworld/configs/eval_config.yaml",
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
    parser.add_argument("--base-url", default="http://localhost:8000/v1/", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError("Config file not found: {}".format(args.config))

    if args.progress_every <= 0:
        raise ValueError("--progress-every must be > 0")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    random.seed(args.seed)

    config = load_config(args.config)
    env_type = config["env"]["type"]
    print("Environment type: {}".format(env_type))
    print("Eval split: {}".format(args.split))
    print("Fixed seed: {}".format(args.seed))
    print("Concurrency: {}".format(args.concurrency))

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
                        args.base_url,
                        args.api_key,
                        args.model,
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
