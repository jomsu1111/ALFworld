#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from alfworld_utils import load_config, set_seed, validate_config_paths
from env_runner import build_env, run_episodes
from oepnai.model_client_gpt5 import get_gpt5_action_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALFWorld ReAct+Reflexion runner with GPT-5 API.")
    parser.add_argument("--config", default="configs/alfworld_llama3_zeroshot.yaml")
    parser.add_argument("--split", choices=["eval_id", "eval_ood"], default="eval_id")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="gpt-5")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI API key. If omitted, uses OPENAI_API_KEY env.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--prompting-mode", choices=["react", "direct"], default="react")
    parser.add_argument("--use-reflexion", dest="use_reflexion", action="store_true")
    parser.add_argument("--no-reflexion", dest="use_reflexion", action="store_false")
    parser.set_defaults(use_reflexion=True)
    parser.add_argument("--reflection-window", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config)
    validate_config_paths(config)
    config["dataset"]["num_eval_games"] = args.episodes
    config["general"]["training_method"] = "dqn"
    config["rl"]["training"]["max_nb_steps_per_episode"] = args.max_steps
    config["env"]["type"] = "AlfredTWEnv"
    config["general"]["use_cuda"] = bool(torch.cuda.is_available())

    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError("No OpenAI API key found. Pass --openai-api-key or set OPENAI_API_KEY.")

    env = build_env(config, args.split)
    policy = get_gpt5_action_policy(
        model_id=args.model_id,
        api_key=api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        history_window=args.history_window,
        prompting_mode=args.prompting_mode,
        use_reflexion=args.use_reflexion,
        reflection_window=args.reflection_window,
        reuse=True,
    )

    run_stats = run_episodes(
        env=env,
        policy=policy,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )

    summary = {
        "timestamp": dt.datetime.now().isoformat(),
        "model_id": args.model_id,
        "split": args.split,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "prompting_mode": args.prompting_mode,
        "use_reflexion": args.use_reflexion,
        "reflection_window": args.reflection_window,
        "success_rate": run_stats["successes"] / max(args.episodes, 1),
        "avg_score": run_stats["total_score"] / max(args.episodes, 1),
        "avg_steps": run_stats["total_steps"] / max(args.episodes, 1),
        "results": run_stats["results"],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (
        f"alfworld_{args.split}_{args.model_id.replace('/', '-')}_"
        f"{args.prompting_mode}_{'reflexion' if args.use_reflexion else 'noreflexion'}_"
        f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===== GPT-5 ReAct/Reflexion summary =====")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
