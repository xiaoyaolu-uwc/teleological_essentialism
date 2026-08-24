"""
deployment.py
=============
Model adapter for the deployment pass. Like OpenAIModel but handles the
five-field output schema: {id, tag, extract, reasoning, confidence}.
"""

import json
import os

from archive.prompt_phase.base import ModelAdapter
from labelling.deployment_prompts import DEPLOYMENT_PROMPT_VERSIONS


class DeploymentModel(ModelAdapter):
    def __init__(self, model_name: str, prompt_version: str = "d_v1", batch_size: int = 5):
        if prompt_version not in DEPLOYMENT_PROMPT_VERSIONS:
            raise ValueError(
                f"Unknown deployment prompt version '{prompt_version}'. "
                f"Available: {list(DEPLOYMENT_PROMPT_VERSIONS)}"
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self._prompts = DEPLOYMENT_PROMPT_VERSIONS[prompt_version]

    def classify(self, texts: list[str]) -> list[dict]:
        results = []
        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start: batch_start + self.batch_size]
            batch_results = self._classify_batch(batch_texts)
            results.extend(batch_results)
        return results

    def _classify_batch(self, texts: list[str]) -> list[dict]:
        import openai

        passages_block = "\n".join(
            self._prompts["passage_template"].format(id=i, text=text[:1500])
            for i, text in enumerate(texts)
        )
        user_msg = self._prompts["user_template"].format(
            n=len(texts), passages_block=passages_block
        )

        client = openai.OpenAI()
        try:
            response = client.responses.create(
                model=self.model_name,
                instructions=self._prompts["system"],
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

        # Strip markdown code fences if present
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
                try:
                    conf = float(r.get("confidence", 0.5))
                    conf = max(0.0, min(1.0, conf))
                except (TypeError, ValueError):
                    conf = 0.5
                out.append({
                    "tag":        r.get("tag", "error").strip().lower(),
                    "extract":    r.get("extract", ""),
                    "reasoning":  r.get("reasoning", ""),
                    "confidence": conf,
                })
            else:
                out.append({
                    "tag": "error", "extract": "", "reasoning": "No result returned", "confidence": 0.0
                })
        return out
