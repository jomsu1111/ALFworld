#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
from pathlib import Path

import torch
from alfworld_utils import (
    load_config,
    set_seed,
    validate_config_paths,
)
from env_runner import build_env, run_episodes
from model_client import get_llama_action_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALFWorld ReAct+Reflexion runner with Llama-3-8B.")
    parser.add_argument("--config", default="configs/alfworld_llama3_zeroshot.yaml")
    parser.add_argument("--split", choices=["eval_id", "eval_ood"], default="eval_id")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. If omitted, uses HF_TOKEN env.")
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
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

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError("No Hugging Face token found. Pass --hf-token or set HF_TOKEN.")

    env = build_env(config, args.split)

    policy = get_llama_action_policy(
        model_id=args.model_id,
        hf_token=hf_token,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
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
        f"alfworld_{args.split}_{args.model_id.split('/')[-1]}_"
        f"{args.prompting_mode}_{'reflexion' if args.use_reflexion else 'noreflexion'}_"
        f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===== ReAct/Reflexion summary =====")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
