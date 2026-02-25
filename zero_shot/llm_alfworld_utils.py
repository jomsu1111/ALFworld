import json
import os
import re


def safe_first(value, default=None):
    if isinstance(value, list):
        return value[0] if len(value) > 0 else default
    if value is None:
        return default
    return value


def extract_task_from_observation(observation_text):
    marker = "Your task is to: "
    if marker in observation_text:
        task_desc = observation_text.partition(marker)[-1].strip()
        obs_wo_task = observation_text.replace(f"{marker}{task_desc}", "").strip()
        return task_desc, obs_wo_task
    return "", observation_text


def resolve_traj_data_path(gamefile_entry):
    if not gamefile_entry:
        return ""
    if os.path.isdir(gamefile_entry):
        traj_data_path = os.path.join(gamefile_entry, "traj_data.json")
        return traj_data_path if os.path.exists(traj_data_path) else ""
    if os.path.isfile(gamefile_entry):
        parent = os.path.dirname(gamefile_entry)
        traj_data_path = os.path.join(parent, "traj_data.json")
        return traj_data_path if os.path.exists(traj_data_path) else ""
    return ""


def load_traj_context(gamefile_entry):
    traj_data_path = resolve_traj_data_path(gamefile_entry)
    if not traj_data_path:
        return {}

    try:
        with open(traj_data_path, "r", encoding="utf-8") as file:
            traj_data = json.load(file)
    except Exception:
        return {}

    human_task_desc = ""
    anns = traj_data.get("turk_annotations", {}).get("anns", [])
    if len(anns) > 0 and isinstance(anns[0], dict):
        human_task_desc = anns[0].get("task_desc", "")

    templated_task_desc = ""
    template_data = traj_data.get("template", {})
    if isinstance(template_data, dict):
        templated_task_desc = template_data.get("task_desc", "")

    scene = traj_data.get("scene", {})
    pddl_params = traj_data.get("pddl_params", {})

    return {
        "traj_data_path": traj_data_path,
        "task_type": traj_data.get("task_type", ""),
        "task_id": traj_data.get("task_id", ""),
        "human_task_desc": human_task_desc,
        "templated_task_desc": templated_task_desc,
        "scene_type": scene.get("scene_type", ""),
        "scene_num": scene.get("scene_num", ""),
        "pddl_params": {
            "object_target": pddl_params.get("object_target", ""),
            "parent_target": pddl_params.get("parent_target", ""),
            "toggle_target": pddl_params.get("toggle_target", ""),
            "mrecep_target": pddl_params.get("mrecep_target", ""),
            "object_sliced": pddl_params.get("object_sliced", ""),
        },
    }


def build_turn_context(turn_index, observation, infos_dict, previous_action, env_type):
    admissible = safe_first(infos_dict.get("admissible_commands"), default=[])
    won = safe_first(infos_dict.get("won"), default=False)
    gc_sr = safe_first(infos_dict.get("goal_condition_success_rate"), default=None)
    gamefile = safe_first(infos_dict.get("extra.gamefile"), default="")
    expert_plan = safe_first(infos_dict.get("extra.expert_plan"), default=[])

    task_from_obs, obs_without_task = extract_task_from_observation(observation)
    traj_context = load_traj_context(gamefile)

    return {
        "turn_index": turn_index,
        "env_type": env_type,
        "previous_action": previous_action,
        "observation_raw": observation,
        "task_from_observation": task_from_obs,
        "observation_without_task": obs_without_task,
        "admissible_commands": admissible,
        "won": won,
        "goal_condition_success_rate": gc_sr,
        "gamefile": gamefile,
        "expert_plan": expert_plan,
        "trajectory_context": traj_context,
    }


def normalize_model_output_to_command(model_text, admissible):
    if not model_text:
        return ""

    clean = strip_think_tags(model_text)
    if "```" in clean:
        clean = clean.replace("```", "").strip()
    if "\n" in clean:
        clean = clean.split("\n", 1)[0].strip()
    if clean.lower().startswith("action:"):
        clean = clean.split(":", 1)[1].strip()

    if clean in admissible:
        return clean

    lower_to_original = {cmd.lower(): cmd for cmd in admissible}
    if clean.lower() in lower_to_original:
        return lower_to_original[clean.lower()]

    lowered_response = clean.lower()
    for cmd in admissible:
        if cmd.lower() in lowered_response:
            return cmd

    return ""


def strip_think_tags(model_text):
    if not model_text:
        return ""

    clean = model_text.strip()
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL | re.IGNORECASE).strip()

    if not clean:
        return ""

    return clean


def trim_history(messages, keep_last_turns=12):
    fixed_prefix = messages[:1]
    history = messages[1:]
    if len(history) <= keep_last_turns * 2:
        return messages
    return fixed_prefix + history[-keep_last_turns * 2:]


def dump_messages_for_debug(messages):
    print("\n========== FULL LLM PROMPT (messages) ==========")
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print("========== END FULL LLM PROMPT ==========" )
