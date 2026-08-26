#!/usr/bin/env python3
"""Title-level exclusions shared by the BHL and IA pool builders.

Both sources leak the same three things past a subject/title animal match:
botany and mineralogy works, publishing ephemera, and general reference books
whose subtitle happens to mention natural history. A plant or mineral term only
excludes when no strong animal term is also present, so "Cultivated plants and
domestic animals" survives while "The British flora" does not.
"""

import re

STRONG_ANIMAL = re.compile(
    r"\b(zoo\w*|animal|animals|bird|birds|insect|insects|quadruped|quadrupeds|"
    r"fish|fishes|beast|beasts|entomolog\w*|ornitholog\w*|concholog\w*|shell|shells|"
    r"mammal|mammals|reptile|reptiles|serpent|serpents|butterfl\w*|moth|moths|"
    r"lepidopter\w*|coleopter\w*|crustacea\w*|mollusk\w*|mollusc\w*|fauna)\b",
    re.I,
)

PLANT_MINERAL = re.compile(
    r"\b(botan\w*|flora|plants|herbal|herbarium|vegetable statics|gardening|"
    r"planting|horticultur\w*|agricultur\w*|mineralog\w*|geolog\w*)\b",
    re.I,
)

EPHEMERA = re.compile(
    r"(proposals for printing|prospectus|school of arts|pocket library|"
    r"vade-?mecum|encyclop\w*|cyclop\w*|dictionary|almanac|receipts|"
    r"deformity|occult|astrolog\w*)",
    re.I,
)


def excluded(title: str) -> bool:
    """True when the title should be dropped from the scanning pool."""
    if EPHEMERA.search(title):
        return True
    if PLANT_MINERAL.search(title) and not STRONG_ANIMAL.search(title):
        return True
    return False
