"""Example: a custom scorer that judges answers with another hosted model.

Drop-in for the Scorer protocol in evalharness/scoring.py. The judge is just
another ModelClient, so it goes through the same retries/timeouts as the
models under test. Wire it in code (not YAML) so you control the prompt:

    from evalharness.providers import OpenAICompatClient
    judge = JudgeScorer(client=OpenAICompatClient(name="judge", model_id="gpt-4o-mini"),
                        weight=3.0)
    # then pass [judge, *built_ins] to weighted_score()
"""
from __future__ import annotations

from evalharness.providers.base import ModelClient

_PROMPT = """Score this answer 0-10 for correctness and completeness.

Question: {prompt}
Expected points: {reference}
Answer: {text}

Reply with only the integer."""


class JudgeScorer:
    name = "llm_judge"

    def __init__(self, client: ModelClient, weight: float = 1.0) -> None:
        self.client = client
        self.weight = weight

    def score(self, prompt: str, reference: str | None, text: str) -> tuple[float, str]:
        resp = self.client.complete(
            _PROMPT.format(prompt=prompt, reference=reference or "n/a", text=text),
            {"temperature": 0.0, "max_tokens": 4},
        )
        try:
            raw = int("".join(c for c in resp.text if c.isdigit()) or "0")
        except ValueError:
            return 0.0, f"judge returned non-numeric: {resp.text!r}"
        return max(0.0, min(raw / 10.0, 1.0)), f"judge said {raw}/10"
