from policies.few_shot import build_few_shot_policy
from policies.react import build_few_shot_react_policy
from policies.reflexion import ReflexionPolicy
from policies.scoring import ScoringPolicy
from policies.zero_shot import build_zero_shot_policy

METHODS = (
    "zero_shot",
    "few_shot",
    "few_shot_react",
    "few_shot_reflexion",
    "few_shot_react_reflexion",
    "few_shot_react_scoring",
)


def build_policy(method: str, *, scoring_samples: int = 5, **common_kwargs):
    if method == "zero_shot":
        return build_zero_shot_policy(**common_kwargs)

    if method == "few_shot":
        return build_few_shot_policy(**common_kwargs)

    if method == "few_shot_react":
        return build_few_shot_react_policy(**common_kwargs)

    if method == "few_shot_reflexion":
        base = build_few_shot_policy(**common_kwargs)
        return ReflexionPolicy(base)

    if method == "few_shot_react_reflexion":
        base = build_few_shot_react_policy(**common_kwargs)
        return ReflexionPolicy(base)

    if method == "few_shot_react_scoring":
        base = build_few_shot_react_policy(**common_kwargs)
        return ScoringPolicy(base, num_samples=scoring_samples)

    raise ValueError(f"Unsupported method: {method}")
