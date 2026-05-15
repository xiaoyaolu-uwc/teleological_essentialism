"""
prompts.py
==========
Prompt templates for the LLM classification pipeline.

Separated from scanning logic so prompts can be iterated on independently.
New versions go into PROMPT_VERSIONS; scan_passages.py still uses the
module-level constants (v1) directly until refactored.

Contents:
  SYSTEM_PROMPT             — v1 system prompt (module-level, used by scan_passages.py)
  USER_PROMPT_TEMPLATE      — per-batch user message; expects {n} and {passages_block}
  PASSAGE_TEMPLATE          — formats a single passage with author/year metadata
  PASSAGE_TEMPLATE_TEXT_ONLY — text-only variant used by the model adapter (no metadata)
  VALID_TAGS                — the set of tags v1 is allowed to return
  PROMPT_VERSIONS           — registry of versioned prompts keyed by "v1", "v2", ...

Prompt version history
----------------------
v1 — original prompt.
     Problems identified via eval harness (eval/evaluate.py, eval/analyze_errors.py):
     1. divine_teleology triggered by divine attribution alone (God created animals)
        even without a purposive claim (animals exist FOR something). Condition 1
        (purpose) was not enforced independently of condition 2 (divine grounding).
     2. Structural/anatomical language (e.g. "essential features", dissection results)
        triggered internal_essence even for single-specimen observations that make no
        general claim about what the animal is.
     3. Incidental functional language (e.g. "for the good of each", organ described
        as an "instrument" in an argument-from-design passage) triggered
        non_divine_teleology even when the passage was not making a definitional claim
        about animals.
     4. Implicit functional definitions (organ-specialisation passages, ecological
        role descriptions) were dismissed as junk because the model required explicit
        "what is this animal" framing.
     5. Mixed tags unused; "exceedingly rare" framing suppressed them even when
        genuinely warranted.

v2 — two-step decision structure.
     Junk is promoted to a gating criterion (Step 1) rather than a residual.
     The gate is framed as an ontological-position test: does the passage commit
     to a position on what an animal or animal part fundamentally is? A sharp
     heuristic is provided: "could this passage be true regardless of how you
     define an animal? If so → junk."
     Step 2 then disambiguates non_divine_teleology / divine_teleology /
     internal_essence only for passages that have cleared the gate. This
     simultaneously fixes Problems 3 (incidental functional language never
     reaches Step 2) and 4 (implicit functional definitions pass Step 1 and
     reach non_divine_teleology). divine_teleology tightened with both conditions
     stated explicitly and a negative case added (Problem 1). internal_essence
     linked back to Step 1 — single-specimen structural observations fail the gate
     (Problem 2). Mixed tags renamed to mixed_ndt-ie / mixed_dt-ie, "do not appear
     often" replaces "exceedingly rare", examples removed (Problem 5).
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

# Text-only template used by the model adapter (no author/year metadata).
PASSAGE_TEMPLATE_TEXT_ONLY = """\
--- Passage {id} ---
{text}
"""

# ---------------------------------------------------------------------------
# v2 prompt
# ---------------------------------------------------------------------------

VALID_TAGS_V2 = {
    "divine_teleology", "non_divine_teleology", "internal_essence", "junk",
    "mixed_ndt-ie", "mixed_dt-ie",
}

USER_PROMPT_TEMPLATE_V2 = """\
Classify each of the following {n} passages.

You MUST return a JSON array of exactly {n} objects, one per passage, in the \
same order as the passages appear below. Each object MUST have exactly these \
three fields:
  "id"        — the passage id shown in the header (copy it exactly as an integer)
  "tag"       — one of: divine_teleology, non_divine_teleology, internal_essence, \
junk, mixed_ndt-ie, mixed_dt-ie
  "reasoning" — SHORT (under one sentence). Start with your certainty \
(e.g. "Confident:", "Unsure:") then a quick justification pointing to \
specific evidence in the text.

Example output for 2 passages:
[
  {{"id": 0, "tag": "junk", "reasoning": "Confident: describes animal behaviour without committing to what the animal is."}},
  {{"id": 1, "tag": "divine_teleology", "reasoning": "Confident: explicitly claims animals exist for a purpose ordained by God's plan."}}
]

Return ONLY the JSON array. No other text, no wrapper keys.

{passages_block}
"""

SYSTEM_PROMPT_V2 = """\
You are helping classify historical scientific passages about animals for a \
research project studying how animals have been defined across the history of biology.

Your task: apply the following two-step decision to each passage.

────────────────────────────────────────────────────────────
STEP 1 — IS THE PASSAGE DEFINITIONAL? (if no → junk)
────────────────────────────────────────────────────────────

Ask: does this passage commit to an ontological position on what an animal \
or animal part is? The passage must carry an implicit or explicit claim about \
what animals in general, or what a specific kind of animal or part, fundamentally \
are — what makes them the kind of thing they are.

A passage is NOT definitional — and is therefore junk — if it:
- Mentions or describes animals without committing to what they are.
- Describes a process or mechanism without asserting what animals are as \
a consequence of or through that process.
- Uses functional or instrumental language to argue for the existence of \
a creator, rather than to assert what animals exist for.
- Makes an observation about a structural feature of a specimen without \
generalizing to what that kind of animal or part is.
- Catalogs, enumerates, or narrates without characterizing what the animal is.

The sharpest test: could this passage be true regardless of how you define \
an animal? If so, it does not commit to an ontological position and is junk.

If the passage does not clear this bar → tag it junk. Do not proceed to Step 2.

────────────────────────────────────────────────────────────
STEP 2 — IF DEFINITIONAL, WHICH KIND?
────────────────────────────────────────────────────────────

  non_divine_teleology — The passage defines or categorizes an animal or \
part by the function or purpose it serves, without grounding that purpose \
in God. Explicit purposive claims count, but so do implicit ones: an animal \
or part characterized by what it does, what role it plays, or what it is \
suited or adapted for. Explicit definitional framing is not required — \
an implicit functional characterization is sufficient.

  divine_teleology — The passage defines animals or parts as serving a \
purpose AND grounds that purpose in God or a divine plan. Both conditions \
must be independently satisfied:
  (1) The passage claims animals or their parts serve a purpose or exist \
for something.
  (2) That purpose is attributed to divine will, plan, or intellect.
  A passage that attributes animal existence to divine power or wisdom \
without claiming animals were brought about FOR something satisfies only \
condition (2) and is junk. Ask: does the passage say animals serve God’s \
plan, or merely that God made them?

  internal_essence — The passage defines or categorizes animals or parts \
through internal structural or mechanistic features, independently of \
external relationships or functions. The structural feature must be what \
the passage uses to define or categorize the animal — not merely something \
observed in passing. A structural observation that does not generalize to \
what the animal or part is does not pass Step 1 and is junk.

  mixed_ndt-ie — The passage simultaneously defines the animal through \
non-divine function AND through internal structure, with neither being \
incidental to the other. These do not appear often. Use only when genuinely \
confident that both definitional modes are independently present; do not \
use when uncertain between two single tags.

  mixed_dt-ie — The passage simultaneously defines the animal through \
divine purpose AND through internal structure, with neither being incidental. \
These do not appear often. Apply the same double-definitional criterion as \
mixed_ndt-ie.

IMPORTANT GUIDELINES:
- Most passages — probably 70–90% — will be junk.
- The “camp” metadata reflects the AUTHOR’s overall intellectual position, \
not necessarily what THIS specific passage does. An author tagged as \
“divine_teleology” may have plenty of junk passages or even mechanistic \
ones. Judge each passage on its own content, not the author’s camp.
- Do NOT balance the distribution of tags across a batch. Tag each passage \
independently.
"""

# ---------------------------------------------------------------------------
# Versioned prompt registry
# ---------------------------------------------------------------------------
# As prompts evolve, append new entries here (v2, v3, ...) rather than mutating
# existing ones. The eval harness and labelling both select a version at runtime.
# scan_passages.py still imports the module-level constants directly for now.

PROMPT_VERSIONS = {
    "v1": {
        "system":           SYSTEM_PROMPT,
        "user_template":    USER_PROMPT_TEMPLATE,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS,
    },
    "v2": {
        "system":           SYSTEM_PROMPT_V2,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
}
