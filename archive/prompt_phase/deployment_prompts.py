"""
deployment_prompts.py
=====================
Prompt versions for the deployment pass — the run that labels the full
sentences.csv for BERT fine-tuning. These extend the classification system
prompt (taken verbatim from r6a in prompts.py) with a richer output schema:

  id          — passage id (integer)
  tag         — classification label
  extract     — 1–3 verbatim sentences justifying the tag (OCR-corrected at
                word level only); for junk, 1–3 sentences around the animal
                mention (same length policy as non-junk, to avoid giving away
                junk via output length)
  reasoning   — 1–2 sentences explaining why this tag, pointing to evidence
                in the extract and distinguishing from adjacent categories
  confidence  — float 0.0–1.0: probability that the tag is correct

Version history
---------------
d_v1 — initial deployment prompt built on r6a system prompt.
d_v2 — two targeted fixes for format-caused regressions identified in d_v1:
d_v3 — adds hard constraint that extract MUST include the sentence where the
        animal, animals, or animal part is explicitly named. Prevents extracts
        that quote only an abstract principle with no animal referent (e.g.
        "Be it so: this law of growth... is but itself an instrument whereby
        purpose is fulfilled" — no animal named). Accuracy on eval set to
        be verified.
d_v2 — two targeted fixes for format-caused regressions identified in d_v1:
        (1) Tag-anchored extract selection: model now told to pick the extract
            that contains the PRIMARY evidence for the tag it already assigned,
            with per-tag guidance (divine language for DT, functional claim for
            NDT, structural basis for IE, animal-reference sentences for junk).
            Fixes DT→IE confusion when structural language co-occurs with divine
            language (row 6 in d_v1).
        (2) Functionally-defined class reminder in extract instruction: flags
            that for domesticated/parasite/prey/predator passages the class
            label itself is the NDT evidence, so extract should include the
            class-identifying term. Fixes NDT→junk on drooping-ears passage
            (row 7 in d_v1).
        (3) Extract length tightened: "typically 20–60 words" added to
            reduce >80-word extracts (13/49 in d_v1).
"""

# ---------------------------------------------------------------------------
# Import the r6a system prompt as the classification backbone
# ---------------------------------------------------------------------------
from archive.prompt_phase.prompts import PROMPT_VERSIONS as _PV, PASSAGE_TEMPLATE_TEXT_ONLY

_R6A_SYSTEM = _PV["r6a"]["system"]

# ---------------------------------------------------------------------------
# Deployment user template (replaces the eval user template)
# ---------------------------------------------------------------------------

DEPLOYMENT_USER_TEMPLATE_V1 = """\
Classify each of the following {n} passages.

You MUST return a JSON array of exactly {n} objects, one per passage, in the \
same order as the passages appear below. Each object MUST have exactly these \
five fields:

  "id"         — the passage id shown in the header (copy it exactly as an integer)
  "tag"        — one of: divine_teleology, non_divine_teleology, internal_essence, junk
  "extract"    — copy 1–3 consecutive sentences VERBATIM from the passage that \
most directly justify your tag. For junk passages, copy the 1–3 sentences that \
most clearly contain the animal reference (same length as non-junk extracts — \
do NOT use a shorter extract for junk). Correct only obvious OCR character-level \
artifacts where the intended word is unambiguous (e.g. "ufefulnefs" → \
"usefulness", "animais" → "animals", "tlie" → "the") — do NOT rephrase, \
modernize, restructure sentences, or fill in damaged text. If no clean extract \
is possible, quote the best available fragment.
  "reasoning"  — 1–2 sentences explaining why this passage receives this tag. \
Point to specific evidence in your extract and explain why it maps to this \
category rather than an adjacent one (e.g. why it is DT rather than NDT, or \
why it is junk rather than IE).
  "confidence" — your probability (0.0–1.0) that your tag is correct. \
1.0 = certain; 0.5 = genuine toss-up; 0.0 = complete guess. Be well-calibrated: \
if you would be surprised to be wrong, score ≥ 0.85; if the passage is genuinely \
ambiguous between two categories, score ≤ 0.65.

Example output for 2 passages:
[
  {{
    "id": 0,
    "tag": "junk",
    "extract": "The giraffe is found throughout the savannas of sub-Saharan Africa.",
    "reasoning": "Confident: geographic description of where the animal lives; no claim about what the animal is.",
    "confidence": 0.95
  }},
  {{
    "id": 1,
    "tag": "divine_teleology",
    "extract": "Every organ of every animal has been fitted by the Creator to the precise purpose it serves.",
    "reasoning": "Confident: explicitly asserts both that animal parts serve a purpose (condition 1) and that this purpose is divinely ordained (condition 2); not merely claiming God made animals.",
    "confidence": 0.97
  }}
]

Return ONLY the JSON array. No other text, no wrapper keys.

{passages_block}
"""

DEPLOYMENT_USER_TEMPLATE_V2 = """\
Classify each of the following {n} passages.

You MUST return a JSON array of exactly {n} objects, one per passage, in the \
same order as the passages appear below. Each object MUST have exactly these \
five fields:

  "id"         — the passage id shown in the header (copy it exactly as an integer)
  "tag"        — one of: divine_teleology, non_divine_teleology, internal_essence, junk
  "extract"    — copy 1–3 consecutive sentences VERBATIM from the passage that \
contain the PRIMARY evidence for the tag you assigned (typically 20–60 words). \
Choose the extract that would most clearly justify your tag to a reader who \
cannot see your reasoning:
      • divine_teleology: include the sentence(s) with explicit divine or \
purpose-directed language (e.g. "the Creator", "God's plan", "serve a divine \
end") — not just structural or taxonomic language that happens to appear nearby.
      • non_divine_teleology: include the sentence(s) stating the function, \
purpose, or role the animal/part serves. If the passage concerns a \
functionally-defined class (domestic animals, parasites, prey, predators), \
include the sentence containing that class term — the class label itself is \
the NDT evidence.
      • internal_essence: include the sentence(s) asserting structural or \
mechanistic features as the basis for defining or categorising the animal kind.
      • junk: include 1–3 sentences around the main animal reference.
  For all tags: correct only obvious OCR character-level artifacts where the \
intended word is unambiguous (e.g. "ufefulnefs" → "usefulness", "animais" → \
"animals", "tlie" → "the") — do NOT rephrase, modernize, or restructure \
sentences.
  "reasoning"  — 1–2 sentences explaining why this passage receives this tag. \
Point to specific evidence in your extract and explain why it maps to this \
category rather than an adjacent one (e.g. why it is DT rather than NDT, or \
why it is junk rather than IE).
  "confidence" — your probability (0.0–1.0) that your tag is correct. \
1.0 = certain; 0.5 = genuine toss-up; 0.0 = complete guess. Be well-calibrated: \
if you would be surprised to be wrong, score ≥ 0.85; if the passage is genuinely \
ambiguous between two categories, score ≤ 0.65.

Example output for 2 passages:
[
  {{
    "id": 0,
    "tag": "junk",
    "extract": "The giraffe is found throughout the savannas of sub-Saharan Africa.",
    "reasoning": "Geographic description with no claim about what the animal is; fails Step 1.",
    "confidence": 0.95
  }},
  {{
    "id": 1,
    "tag": "divine_teleology",
    "extract": "Every organ of every animal has been fitted by the Creator to the precise purpose it serves.",
    "reasoning": "Asserts both that animal parts serve a purpose and that this purpose is divinely ordained; satisfies both DT conditions.",
    "confidence": 0.97
  }}
]

Return ONLY the JSON array. No other text, no wrapper keys.

{passages_block}
"""

DEPLOYMENT_USER_TEMPLATE_V3 = """\
Classify each of the following {n} passages.

You MUST return a JSON array of exactly {n} objects, one per passage, in the \
same order as the passages appear below. Each object MUST have exactly these \
five fields:

  "id"         — the passage id shown in the header (copy it exactly as an integer)
  "tag"        — one of: divine_teleology, non_divine_teleology, internal_essence, junk
  "extract"    — copy 1–3 consecutive sentences VERBATIM from the passage that \
contain the PRIMARY evidence for the tag you assigned (typically 20–60 words). \
Choose the extract that would most clearly justify your tag to a reader who \
cannot see your reasoning:
      • divine_teleology: include the sentence(s) with explicit divine or \
purpose-directed language (e.g. "the Creator", "God's plan", "serve a divine \
end") — not just structural or taxonomic language that happens to appear nearby.
      • non_divine_teleology: include the sentence(s) stating the function, \
purpose, or role the animal/part serves. If the passage concerns a \
functionally-defined class (domestic animals, parasites, prey, predators), \
include the sentence containing that class term — the class label itself is \
the NDT evidence.
      • internal_essence: include the sentence(s) asserting structural or \
mechanistic features as the basis for defining or categorising the animal kind.
      • junk: include 1–3 sentences around the main animal reference.
  HARD REQUIREMENT (all tags): the extract MUST include at least one sentence \
where an animal, animals, or animal part is explicitly named (e.g. "the \
woodpecker", "animals", "this organ", "the bee's sting"). If the key \
teleological or structural claim appears in a sentence with no animal referent, \
include that sentence AND the nearest sentence that does name the animal or \
part.
  For all tags: correct only obvious OCR character-level artifacts where the \
intended word is unambiguous (e.g. "ufefulnefs" → "usefulness", "animais" → \
"animals", "tlie" → "the") — do NOT rephrase, modernize, or restructure \
sentences.
  "reasoning"  — 1–2 sentences explaining why this passage receives this tag. \
Point to specific evidence in your extract and explain why it maps to this \
category rather than an adjacent one (e.g. why it is DT rather than NDT, or \
why it is junk rather than IE).
  "confidence" — your probability (0.0–1.0) that your tag is correct. \
1.0 = certain; 0.5 = genuine toss-up; 0.0 = complete guess. Be well-calibrated: \
if you would be surprised to be wrong, score ≥ 0.85; if the passage is genuinely \
ambiguous between two categories, score ≤ 0.65.

Example output for 2 passages:
[
  {{
    "id": 0,
    "tag": "junk",
    "extract": "The giraffe is found throughout the savannas of sub-Saharan Africa.",
    "reasoning": "Geographic description with no claim about what the animal is; fails Step 1.",
    "confidence": 0.95
  }},
  {{
    "id": 1,
    "tag": "divine_teleology",
    "extract": "Every organ of every animal has been fitted by the Creator to the precise purpose it serves.",
    "reasoning": "Asserts both that animal parts serve a purpose and that this purpose is divinely ordained; satisfies both DT conditions.",
    "confidence": 0.97
  }}
]

Return ONLY the JSON array. No other text, no wrapper keys.

{passages_block}
"""

# ---------------------------------------------------------------------------
# Deployment prompt registry
# ---------------------------------------------------------------------------

DEPLOYMENT_PROMPT_VERSIONS = {
    "d_v1": {
        "system":           _R6A_SYSTEM,
        "user_template":    DEPLOYMENT_USER_TEMPLATE_V1,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       {"divine_teleology", "non_divine_teleology", "internal_essence", "junk"},
    },
    "d_v2": {
        "system":           _R6A_SYSTEM,
        "user_template":    DEPLOYMENT_USER_TEMPLATE_V2,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       {"divine_teleology", "non_divine_teleology", "internal_essence", "junk"},
    },
    "d_v3": {
        "system":           _R6A_SYSTEM,
        "user_template":    DEPLOYMENT_USER_TEMPLATE_V3,
        "passage_template": PASSAGE_TEMPLATE_TEXT_ONLY,
        "valid_tags":       {"divine_teleology", "non_divine_teleology", "internal_essence", "junk"},
    },
}
