"""
two_layer.py
============
Two-layer classification pipeline.

Layer 1 (L1): Binary junk/non-junk gate.
Layer 2 (L2): Category classifier for passages that cleared L1 (non-junk only).

The final tag for a junk passage is "junk".
The final tag for a non-junk passage is whatever L2 returns.

The classify() output matches the interface expected by evaluate.py:
    [{"tag": str, "reasoning": str}, ...]

Additional fields for two-layer analysis are included:
    "l1_tag":       the raw L1 decision ("junk" or "non_junk")
    "l1_reasoning": the L1 reasoning string
"""

import json
import os

from archive.prompt_phase.base import ModelAdapter
from archive.prompt_phase.two_layer_prompts import L1_PROMPT_VERSIONS, L2_PROMPT_VERSIONS


class TwoLayerModel(ModelAdapter):
    def __init__(
        self,
        model_name: str,
        l1_version: str = "l1_v1",
        l2_version: str = "l2_v1",
        batch_size: int = 10,
    ):
        if l1_version not in L1_PROMPT_VERSIONS:
            raise ValueError(f"Unknown L1 version '{l1_version}'. Available: {list(L1_PROMPT_VERSIONS)}")
        if l2_version not in L2_PROMPT_VERSIONS:
            raise ValueError(f"Unknown L2 version '{l2_version}'. Available: {list(L2_PROMPT_VERSIONS)}")
        self.model_name = model_name
        self.batch_size = batch_size
        self._l1 = L1_PROMPT_VERSIONS[l1_version]
        self._l2 = L2_PROMPT_VERSIONS[l2_version]

    def classify(self, texts: list[str]) -> list[dict]:
        """Classify texts through L1 then L2.

        Returns list of dicts with keys: tag, reasoning, l1_tag, l1_reasoning.
        """
        # --- Layer 1: classify all passages ---
        l1_results = self._run_layer(texts, self._l1)

        # --- Collect indices of non-junk passages ---
        non_junk_indices = [i for i, r in enumerate(l1_results) if r["tag"] == "non_junk"]
        non_junk_texts = [texts[i] for i in non_junk_indices]

        # --- Layer 2: classify non-junk passages only ---
        l2_by_position: dict[int, dict] = {}
        if non_junk_texts:
            l2_results = self._run_layer(non_junk_texts, self._l2)
            for pos, orig_idx in enumerate(non_junk_indices):
                l2_by_position[orig_idx] = l2_results[pos]

        # --- Combine ---
        out = []
        for i, l1 in enumerate(l1_results):
            l1_tag = l1["tag"]
            l1_reasoning = l1["reasoning"]
            if l1_tag == "junk":
                out.append({
                    "tag":          "junk",
                    "reasoning":    l1_reasoning,
                    "l1_tag":       "junk",
                    "l1_reasoning": l1_reasoning,
                })
            else:
                l2 = l2_by_position.get(i, {"tag": "error", "reasoning": "L2 result missing"})
                out.append({
                    "tag":          l2["tag"],
                    "reasoning":    l2["reasoning"],
                    "l1_tag":       "non_junk",
                    "l1_reasoning": l1_reasoning,
                })
        return out

    def _run_layer(self, texts: list[str], prompts: dict) -> list[dict]:
        """Run a single layer (L1 or L2) over all texts, in batches."""
        results = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            results.extend(self._classify_batch(batch, prompts))
        return results

    def _classify_batch(self, texts: list[str], prompts: dict) -> list[dict]:
        import openai

        passages_block = "\n".join(
            prompts["passage_template"].format(id=i, text=text[:1500])
            for i, text in enumerate(texts)
        )
        user_msg = prompts["user_template"].format(n=len(texts), passages_block=passages_block)

        client = openai.OpenAI()
        try:
            response = client.responses.create(
                model=self.model_name,
                instructions=prompts["system"],
                input=user_msg,
            )
        except openai.APIStatusError as e:
            raise RuntimeError(f"OpenAI API error {e.status_code}: {e.message}")

        raw = None
        for block in response.output:
            if block.type == "message":
                for part in block.content:
                    if part.type == "output_text":
                        raw = part.text
                        break
            if raw:
                break
        if raw is None:
            raise RuntimeError(f"No output_text in API response: {response}")

        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
            stripped = stripped.rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse model response as JSON: {e}\nRaw:\n{raw}")

        if isinstance(parsed, dict):
            for key in ("results", "passages", "classifications"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
            else:
                parsed = [parsed]

        by_id = {str(r["id"]): r for r in parsed if "id" in r}
        out = []
        for i in range(len(texts)):
            r = by_id.get(str(i)) or (parsed[i] if i < len(parsed) else None)
            if r:
                out.append({
                    "tag":       r.get("tag", "error").strip().lower(),
                    "reasoning": r.get("reasoning", ""),
                })
            else:
                out.append({"tag": "error", "reasoning": "No result returned for this passage"})
        return out
