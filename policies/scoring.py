import re
from typing import Sequence, Tuple


class ScoringPolicy:
    def __init__(self, base_policy, num_samples: int = 5) -> None:
        self.base_policy = base_policy
        self.num_samples = max(1, int(num_samples))

    def _build_scoring_prompt(
        self,
        observation: str,
        admissible_commands: Sequence[str],
        trajectory,
        task_type: str,
        goal_text: str | None,
    ) -> str:
        summary = "(unavailable)"
        if hasattr(self.base_policy, "_summarize_trajectory"):
            try:
                summary = self.base_policy._summarize_trajectory(trajectory)
            except Exception:
                summary = "(unavailable)"

        goal = goal_text or "unknown"
        choices = "\n".join([f"{i + 1}. {cmd}" for i, cmd in enumerate(admissible_commands)])
        return (
            "You are scoring candidate actions for ALFWorld.\n"
            "Pick the single best next action for goal progress.\n"
            "Scoring rubric:\n"
            "- +40 immediate progress toward goal\n"
            "- +25 precondition satisfaction (open/take/go needed next)\n"
            "- +20 feasibility from current state/observation\n"
            "- -30 repeated or no-op behavior with no new evidence\n"
            "- -20 irrelevant detours\n\n"
            f"Task type: {task_type}\n"
            f"Goal: {goal}\n"
            f"Observation:\n{observation}\n\n"
            f"State summary:\n{summary}\n\n"
            f"Candidate actions:\n{choices}\n\n"
            "Output exactly:\n"
            "Reason: <one concise sentence>\n"
            "Action: <single integer>\n"
            "Score: <0-100 integer>"
        )

    def _parse_index_and_reason(self, raw_text: str, num_actions: int) -> Tuple[int | None, str]:
        action_match = re.findall(
            r"(?:^|\n)\s*action\s*:\s*(\d+)\s*$",
            raw_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        reason_match = re.findall(
            r"(?:^|\n)\s*reason\s*:\s*(.+)",
            raw_text,
            flags=re.IGNORECASE,
        )
        reason = reason_match[-1].strip() if reason_match else "Scoring-based choice."
        if action_match:
            idx = int(action_match[-1]) - 1
            if 0 <= idx < num_actions:
                return idx, reason
        return None, reason

    def _parse_score(self, raw_text: str) -> int:
        score_match = re.findall(
            r"(?:^|\n)\s*score\s*:\s*(-?\d+)\s*$",
            raw_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not score_match:
            return 0
        try:
            return max(0, min(100, int(score_match[-1])))
        except ValueError:
            return 0

    def _generate_topk_candidates(
        self,
        observation: str,
        admissible_commands: Sequence[str],
        trajectory,
        task_type: str,
        goal_text: str | None,
    ) -> list[str]:
        candidates: list[str] = []
        max_attempts = max(self.num_samples * 3, self.num_samples)
        attempts = 0

        while len(candidates) < self.num_samples and attempts < max_attempts:
            attempts += 1
            action, _ = self.base_policy.select_action(
                observation=observation,
                admissible_commands=admissible_commands,
                trajectory=trajectory,
                task_type=task_type,
                goal_text=goal_text,
            )
            if action not in candidates:
                candidates.append(action)

        # Ensure at least one valid fallback candidate.
        if not candidates and admissible_commands:
            candidates.append(admissible_commands[0])

        return candidates

    def _score_topk_actions_with_llm(
        self,
        observation: str,
        candidate_actions: Sequence[str],
        trajectory,
        task_type: str,
        goal_text: str | None,
    ) -> Tuple[str, str]:
        best_idx = 0
        best_score = -1
        best_reason = "Scoring-based choice."

        for idx, action in enumerate(candidate_actions):
            prompt = self._build_scoring_prompt(
                observation=observation,
                admissible_commands=candidate_actions,
                trajectory=trajectory,
                task_type=task_type,
                goal_text=goal_text,
            )
            # Bias the evaluator to rate a specific candidate each pass.
            prompt += f"\n\nCandidate to evaluate now: {idx + 1}. {action}\n"
            messages = [
                {"role": "system", "content": "You are a strict action evaluator."},
                {"role": "user", "content": prompt},
            ]
            raw_text = self.base_policy._generate_text(messages)
            chosen_idx, reason = self._parse_index_and_reason(raw_text, len(candidate_actions))
            score = self._parse_score(raw_text)

            # If model scored a different index, down-weight this pass.
            if chosen_idx is not None and chosen_idx != idx:
                score = max(0, score - 15)

            if score > best_score:
                best_idx = idx
                best_score = score
                best_reason = reason

        return candidate_actions[best_idx], best_reason

    def select_action(
        self,
        observation: str,
        admissible_commands,
        trajectory,
        task_type: str = "pick_and_place_simple",
        goal_text: str | None = None,
    ):
        if hasattr(self.base_policy, "_generate_text"):
            candidates = self._generate_topk_candidates(
                observation=observation,
                admissible_commands=admissible_commands,
                trajectory=trajectory,
                task_type=task_type,
                goal_text=goal_text,
            )
            return self._score_topk_actions_with_llm(
                observation=observation,
                candidate_actions=candidates,
                trajectory=trajectory,
                task_type=task_type,
                goal_text=goal_text,
            )

        # Fallback when base policy is not an LLM policy.
        return self.base_policy.select_action(
            observation=observation,
            admissible_commands=admissible_commands,
            trajectory=trajectory,
            task_type=task_type,
            goal_text=goal_text,
        )
