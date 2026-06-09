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
}
# mixed_ndt-ie and mixed_dt-ie collapsed into non_divine_teleology and divine_teleology
# respectively in the eval set (June 2026). VALID_TAGS_V2 updated to match.

USER_PROMPT_TEMPLATE_V2 = """\
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
# Iteration 1 variants (based on v2 failure analysis)
# ---------------------------------------------------------------------------

# v3a: Extends divine_teleology to capture implicit/aesthetic divine purpose.
# Targets: divine_teleology→junk misses (14/45 in v2).
SYSTEM_PROMPT_V3A = SYSTEM_PROMPT_V2.replace(
    """\
  divine_teleology — The passage defines animals or parts as serving a \
purpose AND grounds that purpose in God or a divine plan. Both conditions \
must be independently satisfied:
  (1) The passage claims animals or their parts serve a purpose or exist \
for something.
  (2) That purpose is attributed to divine will, plan, or intellect.
  A passage that attributes animal existence to divine power or wisdom \
without claiming animals were brought about FOR something satisfies only \
condition (2) and is junk. Ask: does the passage say animals serve God's \
plan, or merely that God made them?""",
    """\
  divine_teleology — The passage defines animals or parts as serving a \
purpose AND grounds that purpose in God or a divine plan. Both conditions \
must be independently satisfied:
  (1) The passage claims animals or their parts serve a purpose or exist \
for something. This condition can be satisfied implicitly: a passage that \
attributes species or animals to a divine plan whose ends include beauty, \
harmony, or variety — even without using the word "purpose" — satisfies \
condition (1), because it is asserting that animals exist to embody or \
produce a divine end.
  (2) That purpose is attributed to divine will, plan, intellect, or \
aesthetic intention.
  A passage that attributes animal existence to divine power or wisdom \
without any purposive or teleological framing satisfies only condition (2) \
and is junk. Ask: does the passage commit to animals being brought about \
for some divine end, or merely that a divine power caused them to exist?"""
)

# v3b: Adds organ-function vs. mechanism-function discriminator to non_divine_teleology.
# Targets: non_divine_teleology→junk misses (18/65 in v2).
SYSTEM_PROMPT_V3B = SYSTEM_PROMPT_V2.replace(
    """\
  non_divine_teleology — The passage defines or categorizes an animal or \
part by the function or purpose it serves, without grounding that purpose \
in God. Explicit purposive claims count, but so do implicit ones: an animal \
or part characterized by what it does, what role it plays, or what it is \
suited or adapted for. Explicit definitional framing is not required — \
an implicit functional characterization is sufficient.""",
    """\
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
and counts. A passage that describes how a mechanism acts upon animals \
without characterizing what those animals are does not pass Step 1 and \
is junk. Ask: is the function predicated of the animal/organ, or of \
something acting on it?"""
)

# v3c: Suppresses "essential features" and "adapted to conditions" as surface triggers.
# Targets: junk→internal_essence FP (9/95); internal_essence→non_divine_teleology (5/30).
SYSTEM_PROMPT_V3C = SYSTEM_PROMPT_V2.replace(
    """\
  internal_essence — The passage defines or categorizes animals or parts \
through internal structural or mechanistic features, independently of \
external relationships or functions. The structural feature must be what \
the passage uses to define or categorize the animal — not merely something \
observed in passing. A structural observation that does not generalize to \
what the animal or part is does not pass Step 1 and is junk.""",
    """\
  internal_essence — The passage defines or categorizes animals or parts \
through internal structural or mechanistic features, independently of \
external relationships or functions. The structural feature must be what \
the passage uses to define or categorize the animal — not merely something \
observed in passing. A structural observation that does not generalize to \
what the animal or part is does not pass Step 1 and is junk.
  Watch for surface triggers: phrases like "essential features," \
"adapted to conditions of existence," or "internal organization" do not \
automatically qualify a passage as internal_essence. The passage must be \
using such features as the basis for a general claim about what the animal \
kind is. A passage that notes a structural regularity or typological \
parallel without asserting what the animal fundamentally is → junk."""
)

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
    "v3a": {
        "system":           SYSTEM_PROMPT_V3A,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v3b": {
        "system":           SYSTEM_PROMPT_V3B,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v3c": {
        "system":           SYSTEM_PROMPT_V3C,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v4": {
        # v3c (surface trigger suppression) + v3b's organ-function discriminator
        "system":           SYSTEM_PROMPT_V3C.replace(
            """\
  non_divine_teleology — The passage defines or categorizes an animal or \
part by the function or purpose it serves, without grounding that purpose \
in God. Explicit purposive claims count, but so do implicit ones: an animal \
or part characterized by what it does, what role it plays, or what it is \
suited or adapted for. Explicit definitional framing is not required — \
an implicit functional characterization is sufficient.""",
            """\
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
and counts. A passage that describes how a mechanism acts upon animals \
without characterizing what those animals are does not pass Step 1 and \
is junk. Ask: is the function predicated of the animal/organ, or of \
something acting on it?"""
        ),
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
}

# ---------------------------------------------------------------------------
# Extract v4 system prompt as a named constant for v5 variants to build on
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_V4 = PROMPT_VERSIONS["v4"]["system"]

# ---------------------------------------------------------------------------
# Iteration 2 variants (based on v4 failure analysis)
# ---------------------------------------------------------------------------

# v5a: Strengthen the junk gate's argument-from-design clause.
# Targets: "eye and the hand" junk→NDT misfire (3/3 in v4).
# The model ignores the existing gate item about "functional language to argue
# for a creator." Making the epistemological vs. ontological distinction explicit.
SYSTEM_PROMPT_V5A = SYSTEM_PROMPT_V4.replace(
    "- Uses functional or instrumental language to argue for the existence of \\\na creator, rather than to assert what animals exist for.",
    "- Uses functional or instrumental language to argue for the existence of \\\na creator, rather than to assert what animals exist for. This is the \\\nepistemological vs. ontological distinction: a passage that says 'these \\\norgans PROVE a designer' is making a claim about evidence for God, not \\\nabout what animals are. Even if the organs are described as instruments \\\nor as adapted, the passage is junk if its primary claim is about what \\\nthose organs imply about God's existence rather than about what those \\\norgans fundamentally are."
)

# v5b: Extend divine_teleology to cover "embodies divine thought" pattern.
# Targets: two Agassiz DT passages called junk (3/3 in v4).
# Agassiz's view: natural classification embodies the Creator's thoughts.
# The current definition requires "animals serve a purpose" but misses the
# pattern where animals/nature ARE the expression of divine intellect/plan.
SYSTEM_PROMPT_V5B = SYSTEM_PROMPT_V4.replace(
    "  A passage that attributes animal existence to divine power or wisdom \\\nwithout claiming animals were brought about FOR something satisfies only \\\ncondition (2) and is junk. Ask: does the passage say animals serve God's \\\nplan, or merely that God made them?",
    "  A passage that attributes animal existence to divine power or wisdom \\\nwithout claiming animals were brought about FOR something satisfies only \\\ncondition (2) and is junk. Ask: does the passage say animals serve God's \\\nplan, or merely that God made them?\n  Note: condition (1) can also be satisfied when a passage claims that \\\nthe natural organization of animals — their classification, their \\\narrangement into types — constitutes or embodies divine thought or \\\nintellect. In this case, the animals' nature IS the expression of a \\\ndivine mind, which is a purposive ontological claim even without the \\\nword 'purpose.'"
)

# v5c: Soften the NDT discriminator to allow organ-nature characterizations.
# Targets: organ-specialization NDT passages still called junk (3/3 in v4).
# The current discriminator blocks passages where natural selection is the
# grammatical subject, even when the result is a characterization of the
# organ's nature by its function.
SYSTEM_PROMPT_V5C = SYSTEM_PROMPT_V4.replace(
    "  Key discriminator: the functional claim must be about what the animal or \\\norgan IS or DOES — not about what a mechanism (natural selection, \\\nuse-inheritance, growth) does TO animals. A passage that describes an organ \\\nas specialised for a particular use characterizes the organ by its function \\\nand counts. A passage that describes how a mechanism acts upon animals \\\nwithout characterizing what those animals are does not pass Step 1 and \\\nis junk. Ask: is the function predicated of the animal/organ, or of \\\nsomething acting on it?",
    "  Key discriminator: the functional claim must characterize the animal or \\\norgan — not merely describe a mechanism's operation. A passage whose \\\nprimary content is how a process (natural selection, use-inheritance) \\\nworks in general, without characterizing any specific animal or organ by \\\nits function, does not pass Step 1 and is junk. But if the passage \\\ncharacterizes the nature or identity of an organ by what it is specialised \\\nfor or suited to do — even if a mechanism is what produced that \\\nspecialisation — the organ IS being defined by its function and the \\\npassage counts as non_divine_teleology. The test is whether the passage \\\ncommits to what the organ's nature IS, not what grammatical subject \\\nperforms the action."
)

# ---------------------------------------------------------------------------
# Iteration 3 variants (based on v5 regression analysis)
# ---------------------------------------------------------------------------
# All v5 variants regressed. Hypothesis: prompt too long/complex for gpt-5.4-mini.
# Strategy: simplification over addition.

# v6a: Radical simplification — strip to essentials, trust the model.
# Removes all discriminator paragraphs, "watch for" clauses, surface trigger
# warnings. Keeps: ontological gate + "could this be true" test + brief tags.
SYSTEM_PROMPT_V6A = """\
You are helping classify historical scientific passages about animals for a \
research project studying how animals have been defined across the history of biology.

Your task: apply this two-step decision to each passage.

────────────────────────────────────────────────────────────
STEP 1 — IS THE PASSAGE DEFINITIONAL? (if no → junk)
────────────────────────────────────────────────────────────

Does this passage commit to a position on what an animal or animal part \
fundamentally IS — what makes it the kind of thing it is?

The sharpest test: could this passage be true regardless of how you define \
an animal? If so, it is junk.

Tag as junk if it: mentions or describes animals without characterizing what \
they are; describes a process or mechanism without a resulting claim about \
animal nature; makes observations without generalizing to what the animal \
kind is; catalogs, narrates, or argues without a definitional claim.

────────────────────────────────────────────────────────────
STEP 2 — IF DEFINITIONAL, WHICH KIND?
────────────────────────────────────────────────────────────

  non_divine_teleology — Defines an animal or part by its function or \
purpose, without divine grounding. Both explicit and implicit functional \
characterizations count.

  divine_teleology — Defines animals or parts as serving a purpose AND \
grounds that in God or a divine plan. Both conditions must independently hold: \
(1) a purposive claim about animals, (2) divine attribution of that purpose. \
Attributing animal existence to divine power without a purposive claim \
satisfies only condition (2) and is junk.

  internal_essence — Defines animals or parts through internal structural \
or mechanistic features, independently of external function.

IMPORTANT GUIDELINES:
- Most passages will be junk. Tag each passage independently.
- Judge each passage on its own content, not the author's camp.
- Do NOT balance tag distributions across a batch.
"""

# v6b: v4 + one concise argument-from-design bullet in the junk gate.
# Shorter and more direct than v5a's extended epistemological explanation.
SYSTEM_PROMPT_V6B = SYSTEM_PROMPT_V4.replace(
    "- Catalogs, enumerates, or narrates without characterizing what the \\\nanimal is.",
    "- Catalogs, enumerates, or narrates without characterizing what the \\\nanimal is.\n- Uses features of animals or their organs as evidence that a designer \\\nexists (argument from design), rather than asserting what those animals \\\nor organs exist for."
)

# v6c: Positive-first junk gate — replace the 5 negative exclusions with a
# single concise positive criterion. Less to misapply.
SYSTEM_PROMPT_V6C = SYSTEM_PROMPT_V4.replace(
    """\
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
an animal? If so, it does not commit to an ontological position and is junk.""",
    """\
A passage IS definitional only if it makes a claim that would be false or \
require revision under a different theory of what animals are. A passage \
about how animals behave, how they came to be, what they prove about God, \
or how scientists classify them — without committing to what they \
fundamentally are — is junk regardless of what language it uses.

The sharpest test: could this passage be true regardless of how you define \
an animal? If so, it is junk."""
)

# ---------------------------------------------------------------------------
# Iteration 4 variants (based on v6 analysis)
# ---------------------------------------------------------------------------

# v7a: Structured CoT — require explicit Step 1 answer in reasoning.
# Forces the model to apply the gate before jumping to a tag.
# Uses a modified user template that asks for two-part reasoning.
USER_PROMPT_TEMPLATE_V7A = """\
Classify each of the following {n} passages.

For each passage, apply the two-step decision from your instructions:
Step 1 — Is this definitional? (yes/no + one-phrase reason)
Step 2 — If yes, which tag?

You MUST return a JSON array of exactly {n} objects, one per passage, in the \
same order as the passages appear below. Each object MUST have exactly these \
three fields:
  "id"        — the passage id shown in the header (copy it exactly as an integer)
  "tag"       — one of: divine_teleology, non_divine_teleology, internal_essence, junk
  "reasoning" — Start with "Step 1: [yes/no] — [reason]. Step 2: [tag justification]."

Return ONLY the JSON array. No other text, no wrapper keys.

{passages_block}
"""

# v7b: v4 + argument-from-design bullet with explicit junk redirect.
# Fixes v6b's junk→DT spike by explicitly naming the correct output tag.
SYSTEM_PROMPT_V7B = SYSTEM_PROMPT_V4.replace(
    "- Catalogs, enumerates, or narrates without characterizing what the \\\nanimal is.",
    "- Catalogs, enumerates, or narrates without characterizing what the \\\nanimal is.\n- Uses features of animals or their organs as evidence that a designer \\\nor Creator exists — argument from design. Tag these as junk, not \\\ndivine_teleology: the passage is arguing about God's existence, not \\\ndefining what animals are for."
)

# v7c: v4 + minimal synthetic examples for the three hardest junk patterns.
# Synthetic examples only — no test data or close paraphrases.
# Targets: "divine creation without telos," "argument from design," "mechanism description."
SYSTEM_PROMPT_V7C = SYSTEM_PROMPT_V4.replace(
    "If the passage does not clear this bar → tag it junk. Do not proceed to Step 2.",
    """If the passage does not clear this bar → tag it junk. Do not proceed to Step 2.

Three patterns that consistently fail this test and must be tagged junk:
(a) A passage that says a divine being CAUSED animals to exist, without claiming \
animals were brought about FOR some end. Existence by divine cause ≠ existence for \
a divine purpose.
(b) A passage that uses animal features as EVIDENCE that a designer exists. The claim \
is epistemological (what these features prove about God), not ontological (what the \
animal is).
(c) A passage that describes how a biological process (selection, inheritance, \
development) operates, without the result being a characterization of what any \
specific animal or part IS."""
)

PROMPT_VERSIONS.update({
    "v5a": {
        "system":           SYSTEM_PROMPT_V5A,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v5b": {
        "system":           SYSTEM_PROMPT_V5B,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v5c": {
        "system":           SYSTEM_PROMPT_V5C,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v6a": {
        "system":           SYSTEM_PROMPT_V6A,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v6b": {
        "system":           SYSTEM_PROMPT_V6B,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v6c": {
        "system":           SYSTEM_PROMPT_V6C,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v7a": {
        "system":           SYSTEM_PROMPT_V4,
        "user_template":    USER_PROMPT_TEMPLATE_V7A,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v7b": {
        "system":           SYSTEM_PROMPT_V7B,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v7c": {
        "system":           SYSTEM_PROMPT_V7C,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    # v7d: v2 + ONLY the two changes that demonstrably helped (v3c surface trigger
    # suppression + v4 NDT discriminator). Clean re-derivation from v2 base —
    # no accumulated v3/v5/v6 text that may have introduced interference.
    # Targets: Acontias (junk, fixed by v3c), mechanical weapons (ndt, fixed by v4).
    # v2 baseline: Acontias 1/5, mechanical weapons 5/5.
    "v7d": {
        "system":           SYSTEM_PROMPT_V2
            # Change 1 (from v3c): surface trigger warning in internal_essence
            .replace(
                "  internal_essence — The passage defines or categorizes animals or parts \\\nthrough internal structural or mechanistic features, independently of \\\nexternal relationships or functions. The structural feature must be what \\\nthe passage uses to define or categorize the animal — not merely something \\\nobserved in passing. A structural observation that does not generalize to \\\nwhat the animal or part is does not pass Step 1 and is junk.",
                "  internal_essence — The passage defines or categorizes animals or parts \\\nthrough internal structural or mechanistic features, independently of \\\nexternal relationships or functions. The structural feature must be what \\\nthe passage uses to define or categorize the animal — not merely something \\\nobserved in passing. A structural observation that does not generalize to \\\nwhat the animal or part is does not pass Step 1 and is junk.\n  Watch for surface triggers: phrases like \"essential features,\" \\\n\"adapted to conditions of existence,\" or \"internal organization\" do not \\\nautomatically qualify a passage as internal_essence. The passage must be \\\nusing such features as the basis for a general claim about what the animal \\\nkind is. A passage that notes a structural regularity or typological \\\nparallel without asserting what the animal fundamentally is → junk."
            )
            # Change 2 (from v4): NDT organ-function discriminator
            .replace(
                "  non_divine_teleology — The passage defines or categorizes an animal or \\\npart by the function or purpose it serves, without grounding that purpose \\\nin God. Explicit purposive claims count, but so do implicit ones: an animal \\\nor part characterized by what it does, what role it plays, or what it is \\\nsuited or adapted for. Explicit definitional framing is not required — \\\nan implicit functional characterization is sufficient.",
                "  non_divine_teleology — The passage defines or categorizes an animal or \\\npart by the function or purpose it serves, without grounding that purpose \\\nin God. Explicit purposive claims count, but so do implicit ones: an animal \\\nor part characterized by what it does, what role it plays, or what it is \\\nsuited or adapted for. Explicit definitional framing is not required — \\\nan implicit functional characterization is sufficient.\n  Key discriminator: the functional claim must be about what the animal or \\\norgan IS or DOES — not about what a mechanism (natural selection, \\\nuse-inheritance, growth) does TO animals. A passage that describes an organ \\\nas specialised for a particular use characterizes the organ by its function \\\nand counts. A passage that describes how a mechanism acts upon animals \\\nwithout characterizing what those animals are does not pass Step 1 and \\\nis junk. Ask: is the function predicated of the animal/organ, or of \\\nsomething acting on it?"
            ),
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
})

# ---------------------------------------------------------------------------
# Iteration 5 variants (based on v7c analysis)
# ---------------------------------------------------------------------------
# v7c (74.8%) is the new best. Two problems:
# (1) IE recall collapsed 78%→56%: microscopic-identity passage (3/3 junk) — v7c's
#     junk pattern (c) and the IE "structural regularity → junk" clause are catching
#     passages that assert structural identity across type-members, which IS a
#     definitional IE claim.
# (2) NDT stuck pair (NS specialise organ + two organs modified) — 3/3 junk since v1.
#     v4's NDT discriminator says "organ specialised for a use counts" but the model
#     focuses on "natural selection" as grammatical subject and fires pattern (c).

SYSTEM_PROMPT_V7C = PROMPT_VERSIONS["v7c"]["system"]

# v8a: v7c + IE safeguard for structural-identity-across-type passages.
# Adds an explicit exception to the IE surface-trigger warning: finding that the
# same microstructures recur across specimens of the same type IS a definitional
# claim about what members of that type are — tag internal_essence, not junk.
SYSTEM_PROMPT_V8A = SYSTEM_PROMPT_V7C.replace(
    '  Watch for surface triggers: phrases like "essential features," '
    '"adapted to conditions of existence," or "internal organization" do not '
    'automatically qualify a passage as internal_essence. The passage must be '
    'using such features as the basis for a general claim about what the animal '
    'kind is. A passage that notes a structural regularity or typological '
    'parallel without asserting what the animal fundamentally is → junk.',
    '  Watch for surface triggers: phrases like "essential features," '
    '"adapted to conditions of existence," or "internal organization" do not '
    'automatically qualify a passage as internal_essence. The passage must be '
    'using such features as the basis for a general claim about what the animal '
    'kind is. A passage that notes a structural regularity or typological '
    'parallel without asserting what the animal fundamentally is → junk.\n'
    '  Exception: a passage that asserts structural identity across specimens '
    'of the same type — finding that the same fine structures or anatomical '
    'features recur regardless of habitat or conditions — IS making a '
    'definitional claim. The persistence of structural identity across '
    'conditions implies structure is constitutive of type membership. '
    'Tag these as internal_essence.'
)

# v8b: v7c + NDT discriminator clarified for "natural selection might specialise" pattern.
# The model reads "natural selection might specialise an organ into one for one function"
# as mechanism-description (pattern c) because NS is the grammatical subject.
# Clarification: the grammatical subject does not determine the tag — the result does.
# If the organ ends up characterised by what it does, it IS defined by its function.
SYSTEM_PROMPT_V8B = SYSTEM_PROMPT_V7C.replace(
    '  Key discriminator: the functional claim must be about what the animal or '
    'organ IS or DOES — not about what a mechanism (natural selection, '
    'use-inheritance, growth) does TO animals. A passage that describes an organ '
    'as specialised for a particular use characterizes the organ by its function '
    'and counts. A passage that describes how a mechanism acts upon animals '
    'without characterizing what those animals are does not pass Step 1 and '
    'is junk. Ask: is the function predicated of the animal/organ, or of '
    'something acting on it?',
    '  Key discriminator: the functional claim must be about what the animal or '
    'organ IS or DOES — not about what a mechanism (natural selection, '
    'use-inheritance, growth) does TO animals. A passage that describes an organ '
    'as specialised for a particular use characterizes the organ by its function '
    'and counts — even when natural selection or another process is the '
    'grammatical subject. What matters is the result: if the organ ends up '
    'defined by what it does or what it is for, the passage is '
    'non_divine_teleology. A passage that describes how a mechanism operates '
    'in general, without committing to what any specific organ or animal IS, '
    'does not pass Step 1 and is junk.'
)

# v8c: v7c + both v8a (IE safeguard) and v8b (NDT grammatical-subject clarification).
# Tests whether the two fixes stack without cross-tag interference.
SYSTEM_PROMPT_V8C = SYSTEM_PROMPT_V8A.replace(
    '  Key discriminator: the functional claim must be about what the animal or '
    'organ IS or DOES — not about what a mechanism (natural selection, '
    'use-inheritance, growth) does TO animals. A passage that describes an organ '
    'as specialised for a particular use characterizes the organ by its function '
    'and counts. A passage that describes how a mechanism acts upon animals '
    'without characterizing what those animals are does not pass Step 1 and '
    'is junk. Ask: is the function predicated of the animal/organ, or of '
    'something acting on it?',
    '  Key discriminator: the functional claim must be about what the animal or '
    'organ IS or DOES — not about what a mechanism (natural selection, '
    'use-inheritance, growth) does TO animals. A passage that describes an organ '
    'as specialised for a particular use characterizes the organ by its function '
    'and counts — even when natural selection or another process is the '
    'grammatical subject. What matters is the result: if the organ ends up '
    'defined by what it does or what it is for, the passage is '
    'non_divine_teleology. A passage that describes how a mechanism operates '
    'in general, without committing to what any specific organ or animal IS, '
    'does not pass Step 1 and is junk.'
)

# ---------------------------------------------------------------------------
# Iteration 6 variants (based on v8 analysis)
# ---------------------------------------------------------------------------
# v8 all regressed from v7c. Key learnings:
# - IE exception works but full paragraph costs DT recall. One sentence only.
# - NDT "grammatical subject" fix too broad: breaks NS-never-produce (junk→NDT).
#   Real issue: NDT stuck passages use HYPOTHETICAL framing ("might"), not wrong subject.
# - v9c targets the 2 Agassiz DT→junk passages via minimal DT extension.

# v9a: v7c + one-sentence IE exception (structural identity across type = IE, not junk)
SYSTEM_PROMPT_V9A = SYSTEM_PROMPT_V7C.replace(
    'A passage that notes a structural regularity or typological '
    'parallel without asserting what the animal fundamentally is → junk.',
    'A passage that notes a structural regularity or typological '
    'parallel without asserting what the animal fundamentally is → junk. '
    'Exception: finding that the same fine structures recur identically '
    'across specimens of the same type, regardless of conditions, IS a '
    'definitional claim about type membership — tag internal_essence.'
)

# v9b: v7c + NDT hypothetical-framing fix
# The stuck NDT passages use "might": "natural selection might easily specialise
# a part into one for one function alone." The model reads this as speculative,
# not definitional. Fix: hypothetical functional characterization still counts.
SYSTEM_PROMPT_V9B = SYSTEM_PROMPT_V7C.replace(
    '  Key discriminator: the functional claim must be about what the animal or '
    'organ IS or DOES — not about what a mechanism (natural selection, '
    'use-inheritance, growth) does TO animals. A passage that describes an organ '
    'as specialised for a particular use characterizes the organ by its function '
    'and counts. A passage that describes how a mechanism acts upon animals '
    'without characterizing what those animals are does not pass Step 1 and '
    'is junk. Ask: is the function predicated of the animal/organ, or of '
    'something acting on it?',
    '  Key discriminator: the functional claim must be about what the animal or '
    'organ IS or DOES — not about what a mechanism (natural selection, '
    'use-inheritance, growth) does TO animals. A passage that describes an organ '
    'as specialised for a particular use characterizes the organ by its function '
    'and counts — including when the language is conditional or hypothetical '
    '("could be specialised for", "might perform one function alone"): the '
    'functional potential IS the characterization. A passage that describes how '
    'a mechanism operates in general, without committing to any organ\'s function, '
    'does not pass Step 1 and is junk.'
)

# v9c: v7c + minimal DT extension for "natural classification embodies divine intellect"
# Targets: 2 Agassiz DT passages consistently called junk (3/3 in v7c).
# These claim that natural scientific classification IS the expression of divine thought.
# Current DT requires animals "serve a purpose" — this pattern satisfies that implicitly.
SYSTEM_PROMPT_V9C = SYSTEM_PROMPT_V7C.replace(
    '  A passage that attributes animal existence to divine power or wisdom '
    'without claiming animals were brought about FOR something satisfies only '
    'condition (2) and is junk. Ask: does the passage say animals serve God’s '
    'plan, or merely that God made them?',
    '  A passage that attributes animal existence to divine power or wisdom '
    'without claiming animals were brought about FOR something satisfies only '
    'condition (2) and is junk. Ask: does the passage say animals serve God’s '
    'plan, or merely that God made them?\n'
    '  Note: condition (1) is also satisfied when a passage claims that the '
    'natural classification or arrangement of animals constitutes or embodies '
    'divine thought — animals existing AS the expression of divine intellect '
    'is a purposive claim.'
)

# ---------------------------------------------------------------------------
# Iteration 7 variants (based on v9 analysis)
# ---------------------------------------------------------------------------
# v9 all failed. Key learnings:
# - IE exception (any form) consistently causes DT→mixed_dt-ie interference. STOP.
# - v9b's hypothetical NDT fix gave FIRST EVER movement on stuck NDT pair (2/3 instead
#   of 3/3), but also made NS-never-produce go NDT (wrong). Fix was too broad.
# - v9c's DT extension right direction (Agassiz passages) but caught 5-6 junk→DT FPs.
#   Junk FPs are classification-system passages, not animal-characterization passages.
#
# v10a: v7c + narrow NDT fix: "organ narrowed/perfected to one specific function → NDT"
#   More precise than v9b — targets the function-committed-to-organ pattern specifically.
# v10b: v7c + refined DT extension with guard (animals in their nature = DT, not systems).
# v10c: v7c + both v10a and v10b combined.

# v10a: Narrow NDT fix — "organ committed to a specific function" pattern
# v9b was too broad ("functional potential IS the characterization") and caught NS-never-
# produce. v10a adds the organ-specificity test: is the ORGAN committed to a function?
SYSTEM_PROMPT_V10A = SYSTEM_PROMPT_V7C.replace(
    '  Key discriminator: the functional claim must be about what the animal or '
    'organ IS or DOES — not about what a mechanism (natural selection, '
    'use-inheritance, growth) does TO animals. A passage that describes an organ '
    'as specialised for a particular use characterizes the organ by its function '
    'and counts. A passage that describes how a mechanism acts upon animals '
    'without characterizing what those animals are does not pass Step 1 and '
    'is junk. Ask: is the function predicated of the animal/organ, or of '
    'something acting on it?',
    '  Key discriminator: the functional claim must be about what the animal or '
    'organ IS or DOES — not about what a mechanism (natural selection, '
    'use-inheritance, growth) does TO animals. A passage that describes an organ '
    'as specialised for a particular use characterizes the organ by its function '
    'and counts — including when an organ is described as having been narrowed or '
    'perfected to perform one specific function, even if the language is '
    'conditional. Ask: does the passage commit to what a specific organ\'s '
    'function IS? If yes → non_divine_teleology. If the passage instead '
    'describes a constraint on what a process produces in general, without '
    'naming what any organ IS for → junk.'
)

# v10b: Refined DT extension — animals-in-their-nature embody divine thought
# v9c caught Agassiz DT passages (good) but also 5-6 junk→DT FPs: passages about
# classification SYSTEMS revealing divine mind, not about what ANIMALS ARE.
# Guard added: the passage must characterize animals, not classification methodology.
SYSTEM_PROMPT_V10B = SYSTEM_PROMPT_V7C.replace(
    '  A passage that attributes animal existence to divine power or wisdom '
    'without claiming animals were brought about FOR something satisfies only '
    'condition (2) and is junk. Ask: does the passage say animals serve God’s '
    'plan, or merely that God made them?',
    '  A passage that attributes animal existence to divine power or wisdom '
    'without claiming animals were brought about FOR something satisfies only '
    'condition (2) and is junk. Ask: does the passage say animals serve God’s '
    'plan, or merely that God made them?\n'
    '  Note: condition (1) is also satisfied when a passage asserts that '
    'animals themselves — in their nature or arrangement — constitute the '
    'expression of divine thought or intellect. The passage must characterize '
    'what animals ARE, not merely what classification methods reveal about God.'
)

# v10c: Both v10a (NDT) and v10b (DT) combined
SYSTEM_PROMPT_V10C = SYSTEM_PROMPT_V10B.replace(
    '  Key discriminator: the functional claim must be about what the animal or '
    'organ IS or DOES — not about what a mechanism (natural selection, '
    'use-inheritance, growth) does TO animals. A passage that describes an organ '
    'as specialised for a particular use characterizes the organ by its function '
    'and counts. A passage that describes how a mechanism acts upon animals '
    'without characterizing what those animals are does not pass Step 1 and '
    'is junk. Ask: is the function predicated of the animal/organ, or of '
    'something acting on it?',
    '  Key discriminator: the functional claim must be about what the animal or '
    'organ IS or DOES — not about what a mechanism (natural selection, '
    'use-inheritance, growth) does TO animals. A passage that describes an organ '
    'as specialised for a particular use characterizes the organ by its function '
    'and counts — including when an organ is described as having been narrowed or '
    'perfected to perform one specific function, even if the language is '
    'conditional. Ask: does the passage commit to what a specific organ\'s '
    'function IS? If yes → non_divine_teleology. If the passage instead '
    'describes a constraint on what a process produces in general, without '
    'naming what any organ IS for → junk.'
)

# ---------------------------------------------------------------------------
# Iteration 8 variants (based on v10 analysis)
# ---------------------------------------------------------------------------
# v10 all failed. Pattern: rule-based NDT/DT/IE changes exhausted — all cause
# cross-tag interference. New approach for v11:
# - v11a: concrete synthetic EXAMPLE in NDT section for the process+result pattern
#   (never tried: corpus/synthetic examples; all prior fixes were abstract rules)
# - v11b: junk pattern (d) targeting "exhibits thought/plan" pattern (parallelism/
#   gradation passage — stuck 2-3/3 as IE/mixed_dt-ie since v1, never targeted)
# - v11c: both combined

# v11a: v7c + synthetic example for NDT organ specialised through process
# The stuck NDT passages use "natural selection might specialise X into one function."
# The model reads this as mechanism-description because process is foregrounded.
# Adding a concrete example where process+result IS NDT should calibrate the model.
SYSTEM_PROMPT_V11A = SYSTEM_PROMPT_V7C.replace(
    '  Key discriminator: the functional claim must be about what the animal or '
    'organ IS or DOES — not about what a mechanism (natural selection, '
    'use-inheritance, growth) does TO animals. A passage that describes an organ '
    'as specialised for a particular use characterizes the organ by its function '
    'and counts. A passage that describes how a mechanism acts upon animals '
    'without characterizing what those animals are does not pass Step 1 and '
    'is junk. Ask: is the function predicated of the animal/organ, or of '
    'something acting on it?',
    '  Key discriminator: the functional claim must be about what the animal or '
    'organ IS or DOES — not about what a mechanism (natural selection, '
    'use-inheritance, growth) does TO animals. A passage that describes an organ '
    'as specialised for a particular use characterizes the organ by its function '
    'and counts. A passage that describes how a mechanism acts upon animals '
    'without characterizing what those animals are does not pass Step 1 and '
    'is junk. Ask: is the function predicated of the animal/organ, or of '
    'something acting on it?\n'
    '  Example of non_divine_teleology: "The forelegs of the mole, modified '
    'through use, are perfectly adapted for digging." The organ is defined by '
    'its function even though a process of modification is mentioned.'
)

# v11b: v7c + junk pattern (d) for "exhibits thought/plan" passages
# The parallelism/gradation passage: "exhibits thought, and plan, and wisdom."
# Model reads this as IE/mixed_dt-ie because of structural/typological language.
# This is actually an epistemological claim about what patterns PROVE, not what
# animals ARE. Add explicit junk pattern for this.
SYSTEM_PROMPT_V11B = SYSTEM_PROMPT_V7C.replace(
    '(c) A passage that describes how a biological process (selection, inheritance, '
    'development) operates, without the result being a characterization of what any '
    'specific animal or part IS.',
    '(c) A passage that describes how a biological process (selection, inheritance, '
    'development) operates, without the result being a characterization of what any '
    'specific animal or part IS.\n'
    '(d) A passage that says the animal kingdom, or patterns among animals, EXHIBITS '
    'thought, plan, or intelligence — this claims what the patterns PROVE about a '
    'mind, not what animals fundamentally ARE. Tag junk.'
)

# v11c: v7c + both v11a (NDT example) and v11b (exhibits-thought junk pattern)
SYSTEM_PROMPT_V11C = SYSTEM_PROMPT_V11A.replace(
    '(c) A passage that describes how a biological process (selection, inheritance, '
    'development) operates, without the result being a characterization of what any '
    'specific animal or part IS.',
    '(c) A passage that describes how a biological process (selection, inheritance, '
    'development) operates, without the result being a characterization of what any '
    'specific animal or part IS.\n'
    '(d) A passage that says the animal kingdom, or patterns among animals, EXHIBITS '
    'thought, plan, or intelligence — this claims what the patterns PROVE about a '
    'mind, not what animals fundamentally ARE. Tag junk.'
)

# ---------------------------------------------------------------------------
# Iteration 9 variants — FINAL (based on v11 analysis)
# ---------------------------------------------------------------------------
# v11b's "exhibits thought" junk pattern (d) fixed parallelism/gradation (0/3)
# and mechanical weapons (0/3) — first fixes ever for these. But:
# - Pattern too broad: caught DT "mutual dependence...exhibits thought" as junk
# - NS-never-produce regressed to 3/3 NDT
#
# v12a: v7c + refined pattern (d) with God-attribution guard
#   "exhibits thought WITHOUT God attribution → junk; WITH God → Step 2"
# v12b: v7c + refined pattern (d) + explicit junk pattern (e) for NS-never-produce
#   ("selection will never produce X in being Y" = mechanism constraint → junk)
# v12c: v7c + all 5 junk patterns: (a)-(d) refined + (e) NS-never-produce
#   (rewrite of junk patterns block as clean synthesis of all learned patterns)

SYSTEM_PROMPT_V7C_PATTERNS_ANCHOR = (
    '(c) A passage that describes how a biological process (selection, inheritance, '
    'development) operates, without the result being a characterization of what any '
    'specific animal or part IS.'
)

# v12a: refined pattern (d) with God-attribution guard
SYSTEM_PROMPT_V12A = SYSTEM_PROMPT_V7C.replace(
    SYSTEM_PROMPT_V7C_PATTERNS_ANCHOR,
    SYSTEM_PROMPT_V7C_PATTERNS_ANCHOR + '\n'
    '(d) A passage that says animal patterns or the animal kingdom EXHIBITS thought, '
    'plan, or wisdom WITHOUT attributing that thought to God or a divine source — '
    'this claims what patterns prove about a mind, not what animals ARE. Tag junk. '
    '(Where the thought IS attributed to divine design, proceed to Step 2.)'
)

# v12b: refined pattern (d) + pattern (e) for mechanism-constraint passages
SYSTEM_PROMPT_V12B = SYSTEM_PROMPT_V7C.replace(
    SYSTEM_PROMPT_V7C_PATTERNS_ANCHOR,
    SYSTEM_PROMPT_V7C_PATTERNS_ANCHOR + '\n'
    '(d) A passage that says animal patterns EXHIBIT thought or plan WITHOUT '
    'divine attribution → junk. (With divine attribution → Step 2.)\n'
    '(e) A passage that states what a mechanism will or will not produce in '
    'organisms ("selection will never produce X in any being") describes the '
    'mechanism\'s constraint, not what any organism IS — tag junk.'
)

# v12c: full synthesis — all five junk patterns, clean rewrite
SYSTEM_PROMPT_V12C = SYSTEM_PROMPT_V7C.replace(
    'Three patterns that consistently fail this test and must be tagged junk:\n'
    '(a) A passage that says a divine being CAUSED animals to exist, without claiming '
    'animals were brought about FOR some end. Existence by divine cause ≠ existence for '
    'a divine purpose.\n'
    '(b) A passage that uses animal features as EVIDENCE that a designer exists. The claim '
    'is epistemological (what these features prove about God), not ontological (what the '
    'animal is).\n'
    '(c) A passage that describes how a biological process (selection, inheritance, '
    'development) operates, without the result being a characterization of what any '
    'specific animal or part IS.',
    'Five patterns that consistently fail this test and must be tagged junk:\n'
    '(a) Divine cause without divine purpose: "God made them" without "they exist FOR '
    'something."\n'
    '(b) Argument from design: animal features used as EVIDENCE a designer exists.\n'
    '(c) Mechanism description: how selection/inheritance operates without characterizing '
    'what any specific animal or organ IS.\n'
    '(d) "Exhibits thought" without divine attribution: patterns said to exhibit plan or '
    'intelligence, but not attributed to God — epistemological, not ontological.\n'
    '(e) Mechanism constraint: what a process will/will not produce in beings '
    '("selection never produces X") — describes process limits, not what animals ARE.'
)

PROMPT_VERSIONS.update({
    "v12a": {
        "system":           SYSTEM_PROMPT_V12A,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v12b": {
        "system":           SYSTEM_PROMPT_V12B,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v12c": {
        "system":           SYSTEM_PROMPT_V12C,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
})

PROMPT_VERSIONS.update({
    "v11a": {
        "system":           SYSTEM_PROMPT_V11A,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v11b": {
        "system":           SYSTEM_PROMPT_V11B,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v11c": {
        "system":           SYSTEM_PROMPT_V11C,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
})

PROMPT_VERSIONS.update({
    "v10a": {
        "system":           SYSTEM_PROMPT_V10A,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v10b": {
        "system":           SYSTEM_PROMPT_V10B,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v10c": {
        "system":           SYSTEM_PROMPT_V10C,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
})

PROMPT_VERSIONS.update({
    "v9a": {
        "system":           SYSTEM_PROMPT_V9A,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v9b": {
        "system":           SYSTEM_PROMPT_V9B,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v9c": {
        "system":           SYSTEM_PROMPT_V9C,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
})

PROMPT_VERSIONS.update({
    "v8a": {
        "system":           SYSTEM_PROMPT_V8A,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v8b": {
        "system":           SYSTEM_PROMPT_V8B,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
    "v8c": {
        "system":           SYSTEM_PROMPT_V8C,
        "user_template":    USER_PROMPT_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       VALID_TAGS_V2,
    },
})
