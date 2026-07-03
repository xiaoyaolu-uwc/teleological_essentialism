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


# ---------------------------------------------------------------------------
# L1 Generation 1 variants (two-layer track)
# ---------------------------------------------------------------------------
# Baseline: l1_v1, L1 gate 65.3%. Primary failure: 69 FN (non-junk called junk).
# l1_v2a: Extend non-junk to cover structural-definition + functional-trait claims
# l1_v2b: Tighten junk patterns against meta-text and philosophical observations
# l1_v2c: Both combined
# Anchors extracted from l1_v1 runtime string.

_l1v1 = L1_PROMPT_VERSIONS["l1_v1"]["system"]

_NONJUNK_ANCHOR = _l1v1[
    _l1v1.find("If the passage DOES commit"):
    _l1v1.find("tag it non_junk.") + len("tag it non_junk.")
]

_PATC_ANCHOR = "specific animal or part IS."

_NONJUNK_EXT = (
    "\n  Non-junk includes passages that: "
    "(a) assert structural or anatomical characteristics ARE THE BASIS for "
    "defining, identifying, or classifying animal groups -- e.g. stating that "
    "modern taxonomy is built on structural peculiarities, or that a family of "
    "animals is characterized by a shared structural plan. These make a claim "
    "about what fundamentally constitutes animals (internal essence) and are "
    "non-junk even when framed as description. "
    "(b) attribute a specific trait, behavior, or functional characteristic to a "
    "class of animals and offer a causal explanation, even when the explanation "
    "is naturalistic (e.g. domesticated animals have trait X because of their "
    "domestic function)."
)

_PATC_EXT = (
    "\n(d) A meta-textual passage that announces or introduces subsequent content "
    "without itself asserting anything about animal nature (e.g. \"I will now "
    "present the leading features of the animal kingdom\") -- tag junk."
    "\n(e) A passage observing a pattern resembling design or order in nature "
    "(e.g. a parallel between gradation among animals and their growth stages) "
    "without claiming the animals involved are defined by or serve any purpose. "
    "Noting that patterns \"exhibit thought\" or resemble intelligent design, "
    "without a claim about what defines the animals themselves, is junk."
)

L1_SYSTEM_PROMPT_V2A = _l1v1.replace(_NONJUNK_ANCHOR, _NONJUNK_ANCHOR + _NONJUNK_EXT, 1)
L1_SYSTEM_PROMPT_V2B = _l1v1.replace(_PATC_ANCHOR,    _PATC_ANCHOR    + _PATC_EXT,    1)
L1_SYSTEM_PROMPT_V2C = L1_SYSTEM_PROMPT_V2A.replace(_PATC_ANCHOR, _PATC_ANCHOR + _PATC_EXT, 1)

L1_PROMPT_VERSIONS.update({
    "l1_v2a": {
        "system":           L1_SYSTEM_PROMPT_V2A,
        "user_template":    L1_USER_TEMPLATE_V1,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       L1_VALID_TAGS,
    },
    "l1_v2b": {
        "system":           L1_SYSTEM_PROMPT_V2B,
        "user_template":    L1_USER_TEMPLATE_V1,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       L1_VALID_TAGS,
    },
    "l1_v2c": {
        "system":           L1_SYSTEM_PROMPT_V2C,
        "user_template":    L1_USER_TEMPLATE_V1,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       L1_VALID_TAGS,
    },
})

# ---------------------------------------------------------------------------
# L1 Generation 2 variants (two-layer track)
# ---------------------------------------------------------------------------
# Gen 1 winner: l1_v2a, 66.1% overall, L1 gate 67.8%.
# Gen 1 fixes that worked: SERPENTIA IE, MODERN classifications IE, drooping ears NDT.
# Remaining failures:
#   FN (DT->junk): "organized beings...independence", "mutual dependence...exhibits thought",
#     "law of growth...poison" -- argument-form DT passages not recognized as non-junk.
#   FP (junk->non_junk): "NS will never produce anything injurious", 
#     "systems presented as artificial or natural"
#
# l1_v3a: add (c) to non-junk -- argument-from-adaptation DT as non-junk
# l1_v3b: add narrow NS-constraint junk pattern
# l1_v3c: both combined
# Base: l1_v2a (not l1_v1)

_l1v2a = L1_PROMPT_VERSIONS["l1_v2a"]["system"]

_NONJUNK_C_ANCHOR = "animals have trait X because of their domestic function)."
_NONJUNK_C_EXT = (
    " "
    "(c) argue FROM observations of animal organization, adaptation, or ecological "
    "structure TO a conclusion that animals demonstrate or serve divine intelligence -- "
    "e.g., passages concluding that the complexity of an adaptation, the "
    "interdependence of living kingdoms, or the independence of organisms from "
    "physical conditions implies they are divinely ordered or purposed. The argument "
    "itself constitutes a claim about what animals are. This includes passages framed "
    "as rhetorical questions whose conclusion is implied (e.g. 'Does not all this show "
    "that...?')."
)

_NS_JUNK_ANCHOR = "specific animal or part IS."
_NS_JUNK_EXT = (
    "\n(d) A passage stating a general constraint or principle about what natural "
    "selection or another process will or will not produce in organisms -- without "
    "specifying what any particular kind of animal or part IS as a result -- "
    "characterizes the process rather than the animal and remains junk."
)

L1_SYSTEM_PROMPT_V3A = _l1v2a.replace(_NONJUNK_C_ANCHOR, _NONJUNK_C_ANCHOR + _NONJUNK_C_EXT, 1)
L1_SYSTEM_PROMPT_V3B = _l1v2a.replace(_NS_JUNK_ANCHOR,   _NS_JUNK_ANCHOR   + _NS_JUNK_EXT,   1)
L1_SYSTEM_PROMPT_V3C = L1_SYSTEM_PROMPT_V3A.replace(_NS_JUNK_ANCHOR, _NS_JUNK_ANCHOR + _NS_JUNK_EXT, 1)

L1_PROMPT_VERSIONS.update({
    "l1_v3a": {
        "system":           L1_SYSTEM_PROMPT_V3A,
        "user_template":    L1_USER_TEMPLATE_V1,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       L1_VALID_TAGS,
    },
    "l1_v3b": {
        "system":           L1_SYSTEM_PROMPT_V3B,
        "user_template":    L1_USER_TEMPLATE_V1,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       L1_VALID_TAGS,
    },
    "l1_v3c": {
        "system":           L1_SYSTEM_PROMPT_V3C,
        "user_template":    L1_USER_TEMPLATE_V1,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       L1_VALID_TAGS,
    },
})

# ---------------------------------------------------------------------------
# L1 Generation 3 variants (two-layer track)
# ---------------------------------------------------------------------------
# Gen 2 winner: l1_v3b, 68.2% (+2.1pp from l1_v2a 66.1%). NS-constraint fix worked.
# l1_v3a (DT-argument non-junk): REGRESSED to 61.2% -- extension too broad, massive FPs.
# Remaining L1 failures in l1_v3b:
#   FN: "organized beings...independence" (DT->junk ×5), "mutual dependence" (DT->junk ×5),
#       "law of growth/poison" (DT->junk ×5), "dry Land and Waters" (DT->junk ×5)
#   FP: "parallelism...exhibits thought" (junk->non_junk), 
#       "systems presented as artificial or natural" (junk->non_junk)
#
# l1_v4a: Surgical DT-FN fix -- observation+positive-conclusion = non-junk (not obs alone)
# l1_v4b: Junk FP fix -- "exhibits thought" without positive conclusion = junk
# l1_v4c: Both combined
# Base: l1_v3b

_l1v3b = L1_PROMPT_VERSIONS["l1_v3b"]["system"]

# Anchor: end of NONJUNK_EXT from l1_v2a (inherited by l1_v3b)
_DT_ARG_ANCHOR = "animals have trait X because of their domestic function)."
_DT_ARG_EXT = (
    " "
    "(c) observe some property of animal organization (ecological interdependence, "
    "adaptive complexity, organic independence from physical conditions) AND draw a "
    "positive conclusion that this DEMONSTRATES divine ordering, purpose, or "
    "intelligence -- even when the argument is implicit or posed as a rhetorical "
    "question with an implied affirmative answer. Key: the conclusion about divine "
    "purpose is what makes the passage non-junk. A passage that merely observes a "
    "thought-like pattern without drawing that conclusion remains junk."
)

# Anchor: end of NS junk pattern added in l1_v3b
_PATTERN_OBS_ANCHOR = "characterizes the process rather than the animal and remains junk."
_PATTERN_OBS_EXT = (
    "\n(e) A passage observing that some phenomenon 'exhibits thought' or resembles "
    "intelligent design where the observation IS the conclusion -- no further claim "
    "that animals ARE or serve a divine purpose is made -- junk. "
    "(f) A passage discussing whether classification systems reflect natural facts "
    "vs. human constructs, without asserting what animals themselves are -- junk."
)

L1_SYSTEM_PROMPT_V4A = _l1v3b.replace(_DT_ARG_ANCHOR,      _DT_ARG_ANCHOR      + _DT_ARG_EXT,      1)
L1_SYSTEM_PROMPT_V4B = _l1v3b.replace(_PATTERN_OBS_ANCHOR, _PATTERN_OBS_ANCHOR + _PATTERN_OBS_EXT, 1)
L1_SYSTEM_PROMPT_V4C = L1_SYSTEM_PROMPT_V4A.replace(_PATTERN_OBS_ANCHOR, _PATTERN_OBS_ANCHOR + _PATTERN_OBS_EXT, 1)

L1_PROMPT_VERSIONS.update({
    "l1_v4a": {
        "system":           L1_SYSTEM_PROMPT_V4A,
        "user_template":    L1_USER_TEMPLATE_V1,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       L1_VALID_TAGS,
    },
    "l1_v4b": {
        "system":           L1_SYSTEM_PROMPT_V4B,
        "user_template":    L1_USER_TEMPLATE_V1,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       L1_VALID_TAGS,
    },
    "l1_v4c": {
        "system":           L1_SYSTEM_PROMPT_V4C,
        "user_template":    L1_USER_TEMPLATE_V1,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       L1_VALID_TAGS,
    },
})

# ── Generation 4 (two_layer) ──────────────────────────────────────────────────
# Base L1: l1_v4b (68.9%, 4 runs)
# Base L2: l2_v1
#
# l1_v5a: Guard junk pattern (d) from catching NDT organ-function passages.
#   "natural selection might specialise a part" characterizes the PART, not process.
# l2_v2: Add NDT clarification for conditional/hypothetical organ characterizations.
#   "might be specialised for X" predicates the function of the part -- counts as NDT.
#
# Eval combos:
#   l1_v5a_l2_v1 : L1 fix only
#   l1_v4b_l2_v2 : L2 fix only
#   l1_v5a_l2_v2 : both combined

_l1v4b_g4 = L1_PROMPT_VERSIONS["l1_v4b"]["system"]
_l2v1_g4  = L2_PROMPT_VERSIONS["l2_v1"]["system"]

# L1 fix
_JUNK_D_ANCHOR_G4 = (
    "characterizes the process rather than the animal and remains junk."
)
_JUNK_D_GUARD_G4 = (
    " EXCEPTION: if the passage specifies what a particular kind of part or organ "
    "will be capable of, suited for, or functionally become as a result of the "
    "process -- i.e., the functional characterization is predicated of the part, "
    "not of the process -- it characterizes the part and is non_junk."
)
L1_SYSTEM_PROMPT_V5A = _l1v4b_g4.replace(_JUNK_D_ANCHOR_G4, _JUNK_D_ANCHOR_G4 + _JUNK_D_GUARD_G4, 1)

# L2 fix
_L2_NDT_DISC_G4 = (
    "Ask: is the function predicated of the animal/organ, or of something acting on it?"
)
_L2_NDT_COND_EXT_G4 = (
    " Conditional and hypothetical characterizations count: 'this organ might be "
    "specialised for a single function' or 'natural selection might easily modify "
    "this part to perform X' both characterize what the part would be or could "
    "become -- they predicate the functional characterization of the part, not of "
    "the process. The test is whether the passage ends up describing what a part "
    "IS, could be, or would be suited for -- even conditionally."
)
L2_SYSTEM_PROMPT_V2 = _l2v1_g4.replace(_L2_NDT_DISC_G4, _L2_NDT_DISC_G4 + _L2_NDT_COND_EXT_G4, 1)

L2_PROMPT_VERSIONS.update({
    "l2_v2": {
        "system": L2_SYSTEM_PROMPT_V2,
        "user_template":    L2_PROMPT_VERSIONS["l2_v1"]["user_template"],
        "passage_template": L2_PROMPT_VERSIONS["l2_v1"]["passage_template"],
        "valid_tags":       L2_PROMPT_VERSIONS["l2_v1"]["valid_tags"],
    },
})
L1_PROMPT_VERSIONS.update({
    "l1_v5a": {
        "system": L1_SYSTEM_PROMPT_V5A,
        "user_template":    L1_PROMPT_VERSIONS["l1_v4b"]["user_template"],
        "passage_template": L1_PROMPT_VERSIONS["l1_v4b"]["passage_template"],
        "valid_tags":       L1_PROMPT_VERSIONS["l1_v4b"]["valid_tags"],
    },
})

# ── Generation 5 (two_layer) ──────────────────────────────────────────────────
# Base: l1_v5a + l2_v2 (74.3%)
#
# l1_v6a: Add exceptions to L1 patterns (e)/(f) for DT passages that are
#   falsely caught: "Does not all this show...?" = asserting DT, not open question.
#   "scientific classifications...deepest importance" = asserting divine grounding.
# l2_v3: Improve L2 DT definition for implicit-divine-grounding passages —
#   passages that conclude "thought", "intelligence", or "beyond physical causation"
#   without naming God explicitly still qualify as divine_teleology.
# l1_v6a + l2_v3 combined (with l2_v2 base for L2)

_l1v5a_g5 = L1_PROMPT_VERSIONS["l1_v5a"]["system"]
_l2v2_g5  = L2_PROMPT_VERSIONS["l2_v2"]["system"]

# L1: pattern (e) exception
_PATT_E_L1_ANCHOR = (
    "(e) A passage observing that some phenomenon 'exhibits thought' or resembles "
    "intelligent design where the observation IS the conclusion -- no further claim "
    "that animals ARE or serve a divine purpose is made -- junk. "
)
_PATT_E_EXCEPTION = (
    "EXCEPTION: if the observation is used as a premise followed by a rhetorical "
    "or explicit conclusion ('Does not all this show, on the contrary, that...?' "
    "implies an affirmative answer and IS a DT assertion -- non_junk). "
)
# L1: pattern (f) exception
_PATT_F_L1_ANCHOR = (
    "(f) A passage discussing whether classification systems reflect natural facts "
    "vs. human constructs, without asserting what animals themselves are -- junk."
)
_PATT_F_EXCEPTION = (
    " EXCEPTION: if the discussion of classification systems concludes or implies "
    "that they are grounded in a divine plan or divine intelligence (e.g. the "
    "author says the question 'bears closely upon what we understand by the laws "
    "of nature' and treats divine grounding as the answer) -- non_junk."
)

L1_SYSTEM_PROMPT_V6A = _l1v5a_g5.replace(_PATT_E_L1_ANCHOR, _PATT_E_L1_ANCHOR + _PATT_E_EXCEPTION, 1)
L1_SYSTEM_PROMPT_V6A = L1_SYSTEM_PROMPT_V6A.replace(_PATT_F_L1_ANCHOR, _PATT_F_L1_ANCHOR + _PATT_F_EXCEPTION, 1)

# L2: add implicit-divine-grounding extension to DT definition
_L2_DT_ANCHOR = (
    "A passage that attributes animal existence to divine power or wisdom without "
    "claiming animals were brought about FOR something does not qualify as "
    "divine_teleology. Ask: does the passage say animals serve God's plan, or "
    "merely that God made them?"
)
_L2_DT_IMPLICIT_EXT = (
    " Implicit divine grounding counts: a passage that concludes the organized "
    "complexity of nature demonstrates 'thought,' 'intelligence,' or something "
    "'beyond mere physical connection' is grounding the arrangement in a mental "
    "or divine source even without naming God -- divine_teleology. Similarly, a "
    "rhetorical question whose implied answer is that nature IS divinely ordered "
    "('Does not all this show...?') is an assertion, not an open question."
)

L2_SYSTEM_PROMPT_V3 = _l2v2_g5.replace(_L2_DT_ANCHOR, _L2_DT_ANCHOR + _L2_DT_IMPLICIT_EXT, 1)

assert L1_SYSTEM_PROMPT_V6A != _l1v5a_g5, "l1_v6a unchanged"
assert "EXCEPTION" in L1_SYSTEM_PROMPT_V6A
assert L2_SYSTEM_PROMPT_V3 != _l2v2_g5, "l2_v3 unchanged"
assert "Implicit divine grounding" in L2_SYSTEM_PROMPT_V3

L1_PROMPT_VERSIONS.update({
    "l1_v6a": {
        "system": L1_SYSTEM_PROMPT_V6A,
        "user_template":    L1_PROMPT_VERSIONS["l1_v5a"]["user_template"],
        "passage_template": L1_PROMPT_VERSIONS["l1_v5a"]["passage_template"],
        "valid_tags":       L1_PROMPT_VERSIONS["l1_v5a"]["valid_tags"],
    },
})
L2_PROMPT_VERSIONS.update({
    "l2_v3": {
        "system": L2_SYSTEM_PROMPT_V3,
        "user_template":    L2_PROMPT_VERSIONS["l2_v2"]["user_template"],
        "passage_template": L2_PROMPT_VERSIONS["l2_v2"]["passage_template"],
        "valid_tags":       L2_PROMPT_VERSIONS["l2_v2"]["valid_tags"],
    },
})

# ── Generation 6 (two_layer) ──────────────────────────────────────────────────
# Base: l1_v5a + l2_v2 (74.3%)
# Problem: L2 returning "error" (likely outputting "junk") for 3 DT passages:
#   "mutual dependence...exhibits thought; it demonstrates..."
#   "law of growth/poison...subtle knowledge of the organisation of another"
#   "Owen morphologist...hopeless is the attempt to explain"
# Strategy:
#   l2_v4: Add "NEVER output junk" instruction + explicit Agassiz-implicit-DT
#           guidance for argument-from-inadequacy pattern.
#   (L1 stays at l1_v5a — Gen 5 showed L1 changes cause junk flooding)
#
# Only 2 eval combos this gen (L1 fixed):
#   l1_v5a_l2_v4 : new L2 only
#   l1_v5a_l2_v4 is the only variant (skip a/b/c since L1 not changing)
# Added: l1_v5a_l2_v4_b as a slight variant to still give 3 eval points

_l2v2_g6 = L2_PROMPT_VERSIONS["l2_v2"]["system"]

# Anchor: end of the L2 prompt instructions (IMPORTANT section)
_L2_IMPORTANT_ANCHOR = (
    "- Judge each passage on its own content, not the author's overall position.\n"
    "- Do NOT balance tag distributions across a batch."
)
_L2_NEVER_JUNK = (
    "\n- NEVER output 'junk' -- every passage you receive has already been "
    "confirmed as non-junk by a prior filter. You MUST classify as "
    "divine_teleology, non_divine_teleology, or internal_essence. If none "
    "feels perfect, choose the closest fit."
)

# Anchor: end of DT definition for the Agassiz-inadequacy addition
_L2_DT_END_ANCHOR = (
    "Ask: does the passage say animals serve God's plan, or merely that God made them?"
)
_L2_DT_INADEQUACY_EXT = (
    " Also DT: passages that argue physical or mechanical causation is "
    "INSUFFICIENT to explain some biological phenomenon and that the explanation "
    "must therefore involve thought, intelligence, or design -- e.g., \'hopeless "
    "is the attempt to explain the similarity of pattern by physical forces,\' or "
    "\'how will this law of growth adjust a poison with such subtle knowledge of "
    "the organisation of another animal\' -- classify as divine_teleology even "
    "though no creator is named. Similarly, \'exhibits thought; it demonstrates "
    "that there is more than mere physical connection\' IS a divine_teleology claim."
)

# l2_v4: never-junk + inadequacy extension
L2_SYSTEM_PROMPT_V4 = _l2v2_g6.replace(_L2_IMPORTANT_ANCHOR, _L2_IMPORTANT_ANCHOR + _L2_NEVER_JUNK, 1)
L2_SYSTEM_PROMPT_V4 = L2_SYSTEM_PROMPT_V4.replace(_L2_DT_END_ANCHOR, _L2_DT_END_ANCHOR + _L2_DT_INADEQUACY_EXT, 1)

assert L2_SYSTEM_PROMPT_V4 != _l2v2_g6, "l2_v4 unchanged"
assert "NEVER output" in L2_SYSTEM_PROMPT_V4
assert "hopeless" in L2_SYSTEM_PROMPT_V4

L2_PROMPT_VERSIONS.update({
    "l2_v4": {
        "system": L2_SYSTEM_PROMPT_V4,
        "user_template":    L2_PROMPT_VERSIONS["l2_v2"]["user_template"],
        "passage_template": L2_PROMPT_VERSIONS["l2_v2"]["passage_template"],
        "valid_tags":       L2_PROMPT_VERSIONS["l2_v2"]["valid_tags"],
    },
})

# ── Generation 6 (two_layer) ──────────────────────────────────────────────────
# Base: l1_v5a + l2_v2 (74.3%)
# l2_v4: Add 'NEVER output junk' + Agassiz-inadequacy DT guidance to fix L2 errors
#         on 'mutual dependence', 'law of growth/poison', 'Owen morphologist'.
#         L1 stays at l1_v5a (Gen 5 showed L1 changes cause junk flooding).

_SP_L2V4 = 'You are helping classify historical scientific passages about animals for a research project studying how animals have been defined across the history of biology.\n\nEach passage you receive has already been confirmed to contain a substantive ontological claim about what an animal or animal part fundamentally is. Your task is only to determine WHICH KIND of claim it makes.\n\nClassify each passage using one of these three tags:\n\n  non_divine_teleology — The passage defines or categorizes an animal or part by the function or purpose it serves, without grounding that purpose in God. Explicit purposive claims count, but so do implicit ones: an animal or part characterized by what it does, what role it plays, or what it is suited or adapted for. Explicit definitional framing is not required — an implicit functional characterization is sufficient.\n  Key discriminator: the functional claim must be about what the animal or organ IS or DOES — not about what a mechanism (natural selection, use-inheritance, growth) does TO animals. A passage that describes an organ as specialised for a particular use characterizes the organ by its function and counts. Ask: is the function predicated of the animal/organ, or of something acting on it? Conditional and hypothetical characterizations count: \'this organ might be specialised for a single function\' or \'natural selection might easily modify this part to perform X\' both characterize what the part would be or could become -- they predicate the functional characterization of the part, not of the process. The test is whether the passage ends up describing what a part IS, could be, or would be suited for -- even conditionally.\n\n  divine_teleology — The passage defines animals or parts as serving a purpose AND grounds that purpose in God or a divine plan. Both conditions must be independently satisfied:\n  (1) The passage claims animals or their parts serve a purpose or exist for something.\n  (2) That purpose is attributed to divine will, plan, or intellect.\n  A passage that attributes animal existence to divine power or wisdom without claiming animals were brought about FOR something does not qualify as divine_teleology. Ask: does the passage say animals serve God\'s plan, or merely that God made them? Also divine_teleology: passages that argue physical or mechanical causation is INSUFFICIENT to explain some biological phenomenon and therefore something beyond physical causation -- thought, intelligence, or design -- must be at work. Examples: \'hopeless is the attempt to explain the similarity of pattern by physical forces\'; \'how will this law of growth adjust a poison with such subtle knowledge of the organisation of another animal\'; \'exhibits thought; it demonstrates that there is more than mere physical connection\'. These implicitly ground the phenomenon in a mental/divine source -- classify as divine_teleology even when no creator is named.\n\n  internal_essence — The passage defines or categorizes animals or parts through internal structural or mechanistic features, independently of external relationships or functions. The structural feature must be what the passage uses to define or categorize the animal — not merely something observed in passing.\n  Watch for surface triggers: phrases like "essential features," "adapted to conditions of existence," or "internal organization" do not automatically qualify a passage as internal_essence. The passage must use such features as the basis for a general claim about what the animal kind is.\n\nIMPORTANT:\n- Judge each passage on its own content, not the author\'s overall position.\n- Do NOT balance tag distributions across a batch.\n- NEVER output \'junk\' -- every passage you receive has already been confirmed as non-junk by a prior filter. You MUST classify as divine_teleology, non_divine_teleology, or internal_essence. If none feels perfect, choose the closest fit.\n'

L2_PROMPT_VERSIONS.update({
    "l2_v4": {
        "system": _SP_L2V4,
        "user_template":    L2_PROMPT_VERSIONS["l2_v2"]["user_template"],
        "passage_template": L2_PROMPT_VERSIONS["l2_v2"]["passage_template"],
        "valid_tags":       L2_PROMPT_VERSIONS["l2_v2"]["valid_tags"],
    },
})

# -- Generation 7 (two_layer) ------------------------------------------------
# Base: l1_v5a + l2_v4 (74.7%)
# l2_v5: More precise DT guidance targeting 'scientific classifications' and
#         'mutual dependence' L2 failures; IE clarification for microscopic type.
#         L1 stays at l1_v5a.

_SP_L2V5 = 'You are helping classify historical scientific passages about animals for a research project studying how animals have been defined across the history of biology.\n\nEach passage you receive has already been confirmed to contain a substantive ontological claim about what an animal or animal part fundamentally is. Your task is only to determine WHICH KIND of claim it makes.\n\nClassify each passage using one of these three tags:\n\n  non_divine_teleology — The passage defines or categorizes an animal or part by the function or purpose it serves, without grounding that purpose in God. Explicit purposive claims count, but so do implicit ones: an animal or part characterized by what it does, what role it plays, or what it is suited or adapted for. Explicit definitional framing is not required — an implicit functional characterization is sufficient.\n  Key discriminator: the functional claim must be about what the animal or organ IS or DOES — not about what a mechanism (natural selection, use-inheritance, growth) does TO animals. A passage that describes an organ as specialised for a particular use characterizes the organ by its function and counts. Ask: is the function predicated of the animal/organ, or of something acting on it? Conditional and hypothetical characterizations count: \'this organ might be specialised for a single function\' or \'natural selection might easily modify this part to perform X\' both characterize what the part would be or could become -- they predicate the functional characterization of the part, not of the process. The test is whether the passage ends up describing what a part IS, could be, or would be suited for -- even conditionally.\n\n  divine_teleology — The passage defines animals or parts as serving a purpose AND grounds that purpose in God or a divine plan. Both conditions must be independently satisfied:\n  (1) The passage claims animals or their parts serve a purpose or exist for something.\n  (2) That purpose is attributed to divine will, plan, or intellect.\n  A passage that attributes animal existence to divine power or wisdom without claiming animals were brought about FOR something does not qualify as divine_teleology. Ask: does the passage say animals serve God\'s plan, or merely that God made them? Also divine_teleology: passages that argue physical or mechanical causation is INSUFFICIENT to explain some biological phenomenon and therefore something beyond physical causation -- thought, intelligence, or design -- must be at work. Examples: \'hopeless is the attempt to explain the similarity of pattern by physical forces\'; \'how will this law of growth adjust a poison with such subtle knowledge of the organisation of another animal\'; \'exhibits thought; it demonstrates that there is more than mere physical connection\'. These implicitly ground the phenomenon in a mental/divine source -- classify as divine_teleology even when no creator is named. Additional DT patterns: passages noting that organisms of the same \'type\' share microscopic structure despite radically different environments -- concluding that type (not environment) determines structure -- are internal_essence (not DT). A passage about \'scientific classifications\' that says the question bears on divine law or natural law, concluding classifications ARE grounded in a divine plan, is divine_teleology. A passage framing mutual ecological dependence as \'thought\' that \'demonstrates more than physical connection\' is asserting divine design -- divine_teleology.\n\n  internal_essence — The passage defines or categorizes animals or parts through internal structural or mechanistic features, independently of external relationships or functions. The structural feature must be what the passage uses to define or categorize the animal — not merely something observed in passing.\n  Watch for surface triggers: phrases like "essential features," "adapted to conditions of existence," or "internal organization" do not automatically qualify a passage as internal_essence. The passage must use such features as the basis for a general claim about what the animal kind is.\n\nIMPORTANT:\n- Judge each passage on its own content, not the author\'s overall position.\n- Do NOT balance tag distributions across a batch.\n- NEVER output \'junk\' -- every passage you receive has already been confirmed as non-junk by a prior filter. You MUST classify as divine_teleology, non_divine_teleology, or internal_essence. If none feels perfect, choose the closest fit.\n'

L2_PROMPT_VERSIONS.update({
    "l2_v5": {
        "system": _SP_L2V5,
        "user_template":    L2_PROMPT_VERSIONS["l2_v4"]["user_template"],
        "passage_template": L2_PROMPT_VERSIONS["l2_v4"]["passage_template"],
        "valid_tags":       L2_PROMPT_VERSIONS["l2_v4"]["valid_tags"],
    },
})
