import re
from typing import Sequence, Tuple

FAILURE_PATTERNS = (
    r"you can't",
    r"cannot",
    r"can't",
    r"nothing happens",
    r"not possible",
    r"failed",
    r"is closed",
)


class ReflexionPolicy:
    def __init__(self, base_policy) -> None:
        self.base_policy = base_policy
        self._latest_reflexion = ""

    def _update_reflexion(self, trajectory: Sequence[Tuple[str, str]]) -> None:
        if not trajectory:
            return
        last_action, last_observation = trajectory[-1]
        obs_l = last_observation.lower()
        if any(re.search(pattern, obs_l) for pattern in FAILURE_PATTERNS):
            self._latest_reflexion = (
                f"Avoid repeating '{last_action}' immediately; pick a different admissible action using current evidence."
            )

    def select_action(
        self,
        observation: str,
        admissible_commands,
        trajectory,
        task_type: str = "pick_and_place_simple",
        goal_text: str | None = None,
    ):
        self._update_reflexion(trajectory)

        if self._latest_reflexion:
            augmented_observation = (
                f"{observation}\n\nReflexion note: {self._latest_reflexion}"
            )
        else:
            augmented_observation = observation

        return self.base_policy.select_action(
            observation=augmented_observation,
            admissible_commands=admissible_commands,
            trajectory=trajectory,
            task_type=task_type,
            goal_text=goal_text,
        )
