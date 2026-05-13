"""
prompts.py
==========
Prompt templates for the LLM classification pipeline (scan_passages.py).

Separated from scanning logic so prompts can be iterated on independently.

Contents:
  SYSTEM_PROMPT        — instructs the model on the classification task and taxonomy
  USER_PROMPT_TEMPLATE — per-batch user message; expects {n} and {passages_block}
  PASSAGE_TEMPLATE     — formats a single passage; expects {id}, {author}, {year}, {text}
  VALID_TAGS           — the set of tags the model is allowed to return
"""

# ---------------------------------------------------------------------------
# Valid tags
# ---------------------------------------------------------------------------

VALID_TAGS = {"divine_teleology", "non_divine_teleology", "internal_essence", "junk"}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are helping classify historical scientific passages about animals for a \
research project studying how animals have been defined across the history of biology.

Your task: for each passage, decide whether it contains a substantive \
explanation or definition of what an animal (or animal part) IS, and decide what \
the underlying way in which the author conceptualizes an animal is.

Many passages mention animals but reveal nothing about how the author conceptualizes
them. This is the vast majority of content — categorize these as junk.

For passages that DO contain substantive explanatory content about animals \
or animal parts, classify them using one of these tags:

  non_divine_teleology — An animal/part is defined by the functions it has / the purpose it serves. \
This seldom arises in its explicit form (e.g. "this animal exists to control pests"), although do \
capture explicit claims that 'animals have purpose X' when they come up. But also accept implicit claims \
which define/categorize animals based on any function/purpose they have. A common one is when \
animals are defined/categorized by interaction modes they have with the environment — \
think something like "bees are defined by their making of honey."

  divine_teleology — Animals/their parts are defined by the purpose they serve, and this purpose \
is *specifically divine*. Look for anything that says animals serve some plan of God.

  internal_essence — Animals/parts are defined by some internal structure they have. Notice the core \
distinction between this and the teleological explanations above — both of those define the animal based \
on some external interaction it has with the environment. This is the opposite: the animal is defined \
by some features or mechanisms it has independently of the external environment.

  junk — Passage mentions animals but contains no real explanatory or \
definitional content. Sentences which mention animals in passing, taxonomic catalogs, publication info, \
tables of contents, geographic descriptions, narrative anecdotes without explanatory \
substance, OCR artifacts.

IMPORTANT GUIDELINES:
- Be fairly generous with what counts as "explanatory content" — a passage \
doesn't need to be a philosophical treatise. If it characterizes what an \
animal or animal part is FOR, what it IS structurally, or HOW it works \
mechanistically, that counts.
- But be strict about junk. A passage that just lists species names, \
describes where an animal lives without explaining why, or discusses \
publishing logistics is junk.
- The "camp" metadata attached to each passage reflects the AUTHOR's overall \
intellectual position, not necessarily what THIS specific passage does. An \
author tagged as "divine_teleology" may have plenty of junk passages or \
even mechanistic ones. Judge each passage on its own content.
- Do NOT try to balance the distribution of tags. Most passages SHOULD be \
junk — probably 70-90% of them. The non-junk tags will be rare. Tag each \
passage independently based on what it actually says, not on what other \
passages in the same batch were tagged.
"""

# ---------------------------------------------------------------------------
# User prompt (sent once per batch)
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """\
Classify each of the following {n} passages.

You MUST return a JSON array of exactly {n} objects, one per passage, in the \
same order as the passages appear below. Each object MUST have exactly these \
three fields:
  "id"        — the passage id shown in the header (copy it exactly as an integer)
  "tag"       — one of: divine_teleology, non_divine_teleology, internal_essence, junk
  "reasoning" — SHORT (under one sentence). Start with your certainty \
(e.g. "Confident:", "Unsure:") then a quick justification pointing to \
specific evidence in the text.

Example output for 2 passages:
[
  {{"id": 0, "tag": "junk", "reasoning": "Confident: just lists species names and publication dates."}},
  {{"id": 1, "tag": "divine_teleology", "reasoning": "Confident: 'designed by the Creator for the benefit of' directly invokes divine purpose."}}
]

Return ONLY the JSON array. No other text, no wrapper keys.

{passages_block}
"""

# ---------------------------------------------------------------------------
# Single passage block (inserted into USER_PROMPT_TEMPLATE)
# ---------------------------------------------------------------------------

PASSAGE_TEMPLATE = """\
--- Passage {id} ---
Author: {author} ({year})
Text:
{text}
"""

# ---------------------------------------------------------------------------
# Versioned prompt registry
# ---------------------------------------------------------------------------
# As prompts evolve, append new entries here (v2, v3, ...) rather than mutating
# existing ones. The eval harness selects a version via PROMPT_VERSION.
# scan_passages.py still imports the module-level constants directly for now.

PROMPT_VERSIONS = {
    "v1": {
        "system":           SYSTEM_PROMPT,
        "user_template":    USER_PROMPT_TEMPLATE,
        "passage_template": PASSAGE_TEMPLATE,
        "valid_tags":       VALID_TAGS,
    },
}
