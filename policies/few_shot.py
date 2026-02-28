from model_client import LlamaActionPolicy, get_llama_action_policy


def build_few_shot_policy(**kwargs) -> LlamaActionPolicy:
    return get_llama_action_policy(
        use_few_shot=True,
        use_react_format=False,
        prompting_mode="direct",
        **kwargs,
    )
