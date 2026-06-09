"""
two_layer_prompts.py
====================
Prompt registries for the two-layer classification pipeline.

L1: Binary junk/non-junk gate — does the passage commit to an ontological
    position on what an animal or animal part fundamentally is?

L2: Category classifier — among non-junk passages, which kind of claim is
    being made? (divine_teleology / non_divine_teleology / internal_essence)

Both registries follow the same dict structure as PROMPT_VERSIONS in prompts.py:
    { "system": str, "user_template": str, "passage_template": str, "valid_tags": set }

Version naming:
  L1: l1_v1, l1_v2a, l1_v2b, ...
  L2: l2_v1, l2_v2a, l2_v2b, ...
"""

# ---------------------------------------------------------------------------
# Shared passage template (text only — no author/year metadata)
# ---------------------------------------------------------------------------

PASSAGE_TEMPLATE_TEXT_ONLY = """\
--- Passage {id} ---
{text}
"""

# ---------------------------------------------------------------------------
# L1 — Binary junk gate
# ---------------------------------------------------------------------------
# Derived from v7c Step 1 logic, simplified to binary output.
# Valid outputs: junk | non_junk

L1_VALID_TAGS = {"junk", "non_junk"}

L1_SYSTEM_PROMPT_V1 = """\
You are helping filter historical scientific passages about animals for a \
research project.

Your task: for each passage, decide whether it commits to an ontological \
position on what an animal or animal part fundamentally is.

Ask: does this passage carry an implicit or explicit claim about what animals \
in general, or what a specific kind of animal or part, fundamentally ARE — \
what makes them the kind of thing they are?

A passage does NOT commit to such a position — and is therefore junk — if it:
- Mentions or describes animals without committing to what they are.
- Describes a process or mechanism without asserting what animals are as a \
consequence of or through that process.
- Uses functional or instrumental language to argue for the existence of a \
creator, rather than to assert what animals exist for.
- Makes an observation about a structural feature of a specimen without \
generalizing to what that kind of animal or part is.
- Catalogs, enumerates, or narrates without characterizing what the animal is.

The sharpest test: could this passage be true regardless of how you define \
an animal? If so → junk.

Three patterns that consistently fail this test and must be tagged junk:
(a) A passage that says a divine being CAUSED animals to exist, without \
claiming animals were brought about FOR some end. Existence by divine cause \
≠ existence for a divine purpose.
(b) A passage that uses animal features as EVIDENCE that a designer exists. \
The claim is epistemological (what these features prove about God), not \
ontological (what the animal is).
(c) A passage that describes how a biological process (selection, inheritance, \
development) operates, without the result being a characterization of what \
any specific animal or part IS.

If the passage DOES commit to an ontological position — if it characterizes \
what animals or their parts are, what they exist for, or what makes them the \
kind of thing they are — tag it non_junk.

IMPORTANT:
- Most passages — probably 70–90% — will be junk. Do not over-classify.
- Judge each passage on its own content, not the author's overall position.
- Do NOT balance junk/non_junk counts across a batch.
"""

L1_USER_TEMPLATE_V1 = """\
Classify each of the following {n} passages as junk or non_junk.

You MUST return a JSON array of exactly {n} objects, one per passage, in the \
same order as the passages appear below. Each object MUST have exactly these \
three fields:
  "id"        — the passage id shown in the header (integer)
  "tag"       — one of: junk, non_junk
  "reasoning" — SHORT (under one sentence). Start with "Confident:" or \
"Unsure:" then a quick justification.

Return ONLY the JSON array. No other text.

{passages_block}
"""

# ---------------------------------------------------------------------------
# L2 — Category classifier (operates on non-junk passages only)
# ---------------------------------------------------------------------------
# Derived from v7c Step 2 logic, with Step 1 gate entirely removed.
# Valid outputs: divine_teleology | non_divine_teleology | internal_essence

L2_VALID_TAGS = {"divine_teleology", "non_divine_teleology", "internal_essence"}

L2_SYSTEM_PROMPT_V1 = """\
You are helping classify historical scientific passages about animals for a \
research project studying how animals have been defined across the history of biology.

Each passage you receive has already been confirmed to contain a substantive \
ontological claim about what an animal or animal part fundamentally is. \
Your task is only to determine WHICH KIND of claim it makes.

Classify each passage using one of these three tags:

  non_divine_teleology — The passage defines or categorizes an animal or \
part by the function or purpose it serves, without grounding that purpose \
in God. Explicit purposive claims count, but so do implicit ones: an animal \
or part characterized by what it does, what role it plays, or what it is \
suited or adapted for. Explicit definitional framing is not required — \
an implicit functional characterization is sufficient.
  Key discriminator: the functional claim must be about what the animal or \
organ IS or DOES — not about what a mechanism (natural selection, \
use-inheritance, growth) does TO animals. A passage that describes an organ \
as specialised for a particular use characterizes the organ by its function \
and counts. Ask: is the function predicated of the animal/organ, or of \
something acting on it?

  divine_teleology — The passage defines animals or parts as serving a \
purpose AND grounds that purpose in God or a divine plan. Both conditions \
must be independently satisfied:
  (1) The passage claims animals or their parts serve a purpose or exist \
for something.
  (2) That purpose is attributed to divine will, plan, or intellect.
  A passage that attributes animal existence to divine power or wisdom \
without claiming animals were brought about FOR something does not qualify \
as divine_teleology. Ask: does the passage say animals serve God's plan, \
or merely that God made them?

  internal_essence — The passage defines or categorizes animals or parts \
through internal structural or mechanistic features, independently of \
external relationships or functions. The structural feature must be what \
the passage uses to define or categorize the animal — not merely something \
observed in passing.
  Watch for surface triggers: phrases like "essential features," "adapted \
to conditions of existence," or "internal organization" do not automatically \
qualify a passage as internal_essence. The passage must use such features \
as the basis for a general claim about what the animal kind is.

IMPORTANT:
- Judge each passage on its own content, not the author's overall position.
- Do NOT balance tag distributions across a batch.
"""

L2_USER_TEMPLATE_V1 = """\
Classify each of the following {n} passages.

You MUST return a JSON array of exactly {n} objects, one per passage, in the \
same order as the passages appear below. Each object MUST have exactly these \
three fields:
  "id"        — the passage id shown in the header (integer)
  "tag"       — one of: divine_teleology, non_divine_teleology, internal_essence
  "reasoning" — SHORT (under one sentence). Start with "Confident:" or \
"Unsure:" then a quick justification.

Return ONLY the JSON array. No other text.

{passages_block}
"""

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

L1_PROMPT_VERSIONS = {
    "l1_v1": {
        "system":           L1_SYSTEM_PROMPT_V1,
        "user_template":    L1_USER_TEMPLATE_V1,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       L1_VALID_TAGS,
    },
}

L2_PROMPT_VERSIONS = {
    "l2_v1": {
        "system":           L2_SYSTEM_PROMPT_V1,
        "user_template":    L2_USER_TEMPLATE_V1,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       L2_VALID_TAGS,
    },
}
