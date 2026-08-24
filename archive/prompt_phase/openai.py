import json
import os
from pathlib import Path

from archive.prompt_phase.base import ModelAdapter
from archive.prompt_phase.prompts import PROMPT_VERSIONS


class OpenAIModel(ModelAdapter):
    def __init__(self, model_name: str, prompt_version: str = "v1", batch_size: int = 10):
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"Unknown prompt version '{prompt_version}'. Available: {list(PROMPT_VERSIONS)}")
        self.model_name = model_name
        self.batch_size = batch_size
        self._prompts = PROMPT_VERSIONS[prompt_version]

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
        user_msg = self._prompts["user_template"].format(n=len(texts), passages_block=passages_block)

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

        # Map positional id → {tag, reasoning}; fall back to positional if id missing
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
