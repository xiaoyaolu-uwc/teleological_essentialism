"""
corpus_config.py
================
Shared configuration for the teleological essentialism corpus pipeline.

Provides:
  - PATHS: all data file/directory locations relative to the repo root
  - TEXT_METADATA: filename stem → author/title/year/camp metadata
  - Animal keyword lists and ANIMAL_PATTERN regex
  - Thematic keyword lists and THEMATIC_PATTERN regex
  - Small helper functions used across scripts
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — single source of truth for all data locations
# ---------------------------------------------------------------------------
# REPO_ROOT is two levels up from this file (scripts/ → repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "data_dir":       REPO_ROOT / "Data",
    "clean_texts":    REPO_ROOT / "Data" / "texts" / "clean_texts",
    "raw_texts":      REPO_ROOT / "Data" / "texts" / "raw_texts",
    "passages_csv":   REPO_ROOT / "Data" / "passages.csv",
    "sentences_csv":  REPO_ROOT / "Data" / "sentences.csv",
    "promising_csv":  REPO_ROOT / "Data" / "promising_passages.csv",
    "env_file":       REPO_ROOT / ".env",
}

# ---------------------------------------------------------------------------
# Text metadata
# ---------------------------------------------------------------------------

TEXT_METADATA = {
    "agassiz_essay_on_classification_1859": {
        "author": "Louis Agassiz",
        "title": "Essay on Classification",
        "year": 1859,
        "camp": "A",
        "camp_label": "divine_teleology",
    },
    "argyll_reign_of_law_1867": {
        "author": "Duke of Argyll (George Campbell)",
        "title": "The Reign of Law",
        "year": 1867,
        "camp": "A",
        "camp_label": "divine_teleology",
    },
    "cuvier_animal_kingdom_1831": {
        "author": "Georges Cuvier",
        "title": "The Animal Kingdom",
        "year": 1831,
        "camp": "B",
        "camp_label": "functional_teleology",
    },
    "darwin_origin_of_species_1859": {
        "author": "Charles Darwin",
        "title": "On the Origin of Species",
        "year": 1859,
        "camp": "C_prime",
        "camp_label": "naturalistic_teleology",
    },
    "derham_physico_theology_1713": {
        "author": "William Derham",
        "title": "Physico-Theology",
        "year": 1713,
        "camp": "A",
        "camp_label": "divine_teleology",
    },
    "gray_darwiniana_1876": {
        "author": "Asa Gray",
        "title": "Darwiniana",
        "year": 1876,
        "camp": "A_prime",
        "camp_label": "theistic_evolutionary_teleology",
    },
    "haeckel_history_of_creation_vol1_1876": {
        "author": "Ernst Haeckel",
        "title": "The History of Creation, Vol. 1",
        "year": 1876,
        "camp": "C",
        "camp_label": "mechanistic",
    },
    "haeckel_history_of_creation_vol2_1876": {
        "author": "Ernst Haeckel",
        "title": "The History of Creation, Vol. 2",
        "year": 1876,
        "camp": "C",
        "camp_label": "mechanistic",
    },
    "huxley_mans_place_in_nature_1863": {
        "author": "Thomas Huxley",
        "title": "Evidence as to Man's Place in Nature",
        "year": 1863,
        "camp": "C",
        "camp_label": "mechanistic",
    },
    "kirby_bridgewater_animals_1835_vol1": {
        "author": "William Kirby",
        "title": "On the Creation of Animals (Bridgewater VII), Vol. 1",
        "year": 1835,
        "camp": "A",
        "camp_label": "divine_teleology",
    },
    "kirby_bridgewater_animals_1835_vol2": {
        "author": "William Kirby",
        "title": "On the Creation of Animals (Bridgewater VII), Vol. 2",
        "year": 1835,
        "camp": "A",
        "camp_label": "divine_teleology",
    },
    "lamarck_zoological_philosophy_1914trans": {
        "author": "Jean-Baptiste Lamarck",
        "title": "Zoological Philosophy",
        "year": 1809,
        "camp": "C",
        "camp_label": "mechanistic_quasi_teleological",
    },
    "mivart_genesis_of_species_1871": {
        "author": "St. George Jackson Mivart",
        "title": "On the Genesis of Species",
        "year": 1871,
        "camp": "A_prime",
        "camp_label": "immanent_teleology",
    },
    "owen_on_the_nature_of_limbs_1849": {
        "author": "Richard Owen",
        "title": "On the Nature of Limbs",
        "year": 1849,
        "camp": "B",
        "camp_label": "structuralist_essentialist",
    },
    "paley_natural_theology_1802": {
        "author": "William Paley",
        "title": "Natural Theology",
        "year": 1802,
        "camp": "A",
        "camp_label": "divine_teleology",
    },
    "ray_wisdom_of_god_1691": {
        "author": "John Ray",
        "title": "The Wisdom of God Manifested in the Works of the Creation",
        "year": 1691,
        "camp": "A",
        "camp_label": "divine_teleology",
    },
}

# ---------------------------------------------------------------------------
# Animal detection
# ---------------------------------------------------------------------------

ANIMAL_GENERIC = [
    "animal", "animals", "creature", "creatures", "beast", "beasts",
    "organism", "organisms", "species", "fauna", "brute", "brutes",
    "vertebrate", "vertebrates", "invertebrate", "invertebrates",
    "mammal", "mammals", "quadruped", "quadrupeds",
    "bird", "birds", "fowl", "fowls", "avian",
    "fish", "fishes", "pisces",
    "insect", "insects", "entomology",
    "reptile", "reptiles", "serpent", "serpents", "lizard", "lizards",
    "amphibian", "amphibians", "frog", "frogs", "toad", "toads",
    "mollusc", "molluscs", "mollusk", "mollusks", "shell", "shells",
    "crustacean", "crustaceans", "worm", "worms",
    "predator", "predators", "prey",
]

ANIMAL_SPECIES = [
    "rhinoceros", "elephant", "horse", "horses", "dog", "dogs", "cat", "cats",
    "whale", "whales", "dolphin", "dolphins", "seal", "seals",
    "lion", "lions", "tiger", "tigers", "bear", "bears", "wolf", "wolves",
    "deer", "stag", "ox", "oxen", "cow", "bull", "sheep", "lamb",
    "pig", "swine", "hog",
    "monkey", "monkeys", "ape", "apes", "gorilla", "chimpanzee", "orangutan",
    "eagle", "hawk", "falcon", "owl", "sparrow", "pigeon", "dove",
    "hummingbird", "hummingbirds", "woodpecker", "parrot", "partridge",
    "hen", "cock", "chicken", "turkey", "goose", "geese", "duck", "swan",
    "salmon", "trout", "herring", "cod", "pike",
    "bee", "bees", "wasp", "wasps", "ant", "ants", "beetle", "beetles",
    "butterfly", "butterflies", "moth", "moths", "caterpillar",
    "spider", "spiders", "scorpion", "tick", "mite",
    "crab", "crabs", "lobster", "oyster", "mussel", "snail", "slug",
    "snake", "snakes", "viper", "cobra", "turtle", "tortoise",
    "crocodile", "alligator",
    "bat", "bats", "mole", "moles", "mouse", "mice", "rat", "rats",
    "rabbit", "hare", "squirrel", "beaver", "otter", "badger", "fox",
    "opossum", "possum", "kangaroo",
    "coral", "polyp", "polyps", "sponge", "sponges",
    "leech", "leeches", "parasite", "parasites",
]

ANIMAL_ANATOMY = [
    "wing", "wings", "fin", "fins", "claw", "claws", "talon", "talons",
    "beak", "beaks", "feather", "feathers", "fur", "hide", "scale", "scales",
    "limb", "limbs", "paw", "paws", "hoof", "hooves", "horn", "horns",
    "tail", "tails", "antler", "antlers", "tusk", "tusks",
    "egg", "eggs", "nest", "nests", "larva", "larvae", "pupa", "cocoon",
    "instinct", "instincts", "migration", "hibernate", "hibernation",
    "carnivore", "herbivore", "omnivore",
    "vertebra", "vertebrae", "skeleton", "skull", "bone", "bones",
]

ALL_ANIMAL_KEYWORDS = set(
    w.lower() for w in ANIMAL_GENERIC + ANIMAL_SPECIES + ANIMAL_ANATOMY
)

ANIMAL_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in sorted(ALL_ANIMAL_KEYWORDS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)


def get_animal_keywords(text):
    """Return sorted list of all animal keywords found in text."""
    found = set()
    for match in ANIMAL_PATTERN.finditer(text):
        found.add(match.group(0).lower())
    return sorted(found)


def contains_animal(text):
    """Return True if text contains any animal keyword."""
    return bool(ANIMAL_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Thematic detection
# ---------------------------------------------------------------------------
# A sentence is "thematically relevant" if it contains explanatory or
# definitional language tied to the taxonomy: purpose/teleology, structural
# archetype, definition/essence, mechanism/causation, or divine creation.
# Used to decide which neighboring sentences are worth including as context.

THEMATIC_PURPOSE = [
    "purpose", "purposes", "purposive",
    "designed", "design", "designs",
    "adapted", "adaptation", "adaptations", "adapts",
    "function", "functions", "functional", "functionally",
    "serves", "serve", "serving", "served",
    "utility", "useful", "usefulness", "use of", "uses of",
    "fitted", "fitting", "fitness", "fit for",
    "end of", "ends of", "final cause", "final causes",
    "for the sake", "in order to", "so as to", "so that",
    "means to", "instrument of", "instruments of",
    "contrivance", "contrivances", "contrived",
    "provision", "provisions", "appointed",
    "subserve", "subserves", "subserving",
    "teleological", "teleology",
]

THEMATIC_STRUCTURE = [
    "archetype", "archetypes",
    "homolog", "homologue", "homologous", "homology",
    "morpholog", "morphological", "morphology",
    "structural plan", "body plan", "ground plan",
    "type of", "fundamental type", "ideal type",
    "symmetry", "symmetrical",
    "plan of structure", "plan of organization",
    "structurally", "structural relation",
]

THEMATIC_DEFINITION = [
    "defined as", "defines", "definition of",
    "is essentially", "are essentially",
    "consists in", "consists of",
    "nature of", "the nature of",
    "essence of", "essential character",
    "characterized by", "characteristic of",
    "what is a", "what are",
    "may be described as", "is described as",
    "constitutes", "constitute",
    "properly speaking", "strictly speaking",
    "in the strict sense",
]

THEMATIC_MECHANISM = [
    "caused by", "cause of", "causes of",
    "mechanical", "mechanism", "mechanically",
    "material cause", "efficient cause",
    "physical force", "chemical force",
    "law of", "laws of nature",
    "principle of", "principles of",
    "produced by", "results from",
    "transformation of", "inherited from",
    "natural selection", "struggle for",
]

THEMATIC_DIVINE = [
    "wisdom of god", "creator", "creation of",
    "divine", "providence",
    "works of god", "hand of god",
    "almighty", "deity",
    "beautifully contrived", "admirably adapted",
]

ALL_THEMATIC_PHRASES = (
    THEMATIC_PURPOSE + THEMATIC_STRUCTURE +
    THEMATIC_DEFINITION + THEMATIC_MECHANISM + THEMATIC_DIVINE
)

THEMATIC_PATTERN = re.compile(
    r'(' + '|'.join(re.escape(p) for p in sorted(ALL_THEMATIC_PHRASES, key=len, reverse=True)) + r')',
    re.IGNORECASE
)


def contains_thematic(text):
    """Return True if text contains thematic/explanatory language."""
    return bool(THEMATIC_PATTERN.search(text))
