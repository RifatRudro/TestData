"""Shared constants: class order, label synonyms, anatomical-site mapping."""
from __future__ import annotations

import math
import re

# HAM10000 / ISIC 2018 Task 3 label space (alphabetical, as in HAM10000 'dx').
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CLASS_FULL = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
NUM_CLASSES = len(CLASSES)
MEL_IDX = CLASS_TO_IDX["mel"]

# ISIC 2018 Task 3 ground-truth CSV one-hot columns -> our class codes.
ISIC18_COLS = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AKIEC": "akiec",
               "BKL": "bkl", "DF": "df", "VASC": "vasc"}

# Known problematic image in the ISIC 2018 Task 3 test set (removed in DermaMNIST-E,
# Abhishek et al., Sci Data 2025).
KNOWN_BAD_TEST_IMAGES = {"ISIC_0035068"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Anatomical site: HAM10000 'localization' and ISIC-archive 'anatom_site_general'
# are mapped to one coarse vocabulary so metadata works across datasets.
SITES = ["head/neck", "torso", "upper extremity", "lower extremity",
         "palms/soles", "oral/genital", "unknown"]
_SITE_MAP = {
    "face": "head/neck", "scalp": "head/neck", "ear": "head/neck", "neck": "head/neck",
    "head/neck": "head/neck", "head": "head/neck",
    "abdomen": "torso", "back": "torso", "chest": "torso", "trunk": "torso",
    "torso": "torso", "anterior torso": "torso", "posterior torso": "torso",
    "lateral torso": "torso",
    "upper extremity": "upper extremity", "hand": "upper extremity",
    "lower extremity": "lower extremity", "foot": "lower extremity",
    "acral": "palms/soles", "palms/soles": "palms/soles",
    "genital": "oral/genital", "oral/genital": "oral/genital",
}
SEXES = ["male", "female", "unknown"]
META_DIM = 2 + len(SEXES) + len(SITES)  # age, age_missing, sex one-hot, site one-hot


def coarse_site(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    return _SITE_MAP.get(str(value).strip().lower(), "unknown")


def norm_sex(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    v = str(value).strip().lower()
    return v if v in ("male", "female") else "unknown"


# ---------------------------------------------------------------------------
# Diagnosis synonyms for external datasets (BCN20000, PH2, ISIC-archive exports).
# Returns a class code, or None when the diagnosis is outside the 7-class space
# (e.g. invasive SCC, 'other', 'unknown') and the image should be excluded.
_EXACT = {
    "nv": "nv", "mel": "mel", "bcc": "bcc", "akiec": "akiec", "ak": "akiec",
    "bkl": "bkl", "df": "df", "vasc": "vasc", "scc": None, "unk": None,
    "other": None, "ood": None,
}
_RULES = [  # (regex, class) checked in order; first match wins
    (r"lentigo maligna", "mel"),
    (r"melanoma", "mel"),
    (r"basal cell", "bcc"),
    (r"squamous cell carcinoma in situ|bowen|actinic|intraepithelial", "akiec"),
    (r"squamous cell", None),
    (r"seborrh|solar lentigo|lentigo|lichen|lplk|benign keratosis", "bkl"),
    (r"dermatofibroma", "df"),
    (r"vascular|angioma|angiokeratoma|pyogenic granuloma|hemorrhage|haemorrhage", "vasc"),
    (r"nevus|naevus|nevi|common nevus|atypical nevus", "nv"),
]


def map_diagnosis(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    v = str(value).strip().lower()
    if v in _EXACT:
        return _EXACT[v]
    for pattern, cls in _RULES:
        if re.search(pattern, v):
            return cls
    return None


def _is_excluded(v: str) -> bool:
    v = v.strip().lower()
    return v in ("scc", "other", "unk", "ood") or (re.search(r"squamous cell", v) is not None
                                                   and re.search(r"in situ", v) is None)


def map_diagnosis_row(values):
    """values ordered from most to least specific (e.g. ISIC diagnosis_5 ... diagnosis_2).
    Returns the first mappable class; stops early on an explicit exclusion (invasive SCC, other)."""
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
            continue
        c = map_diagnosis(v)
        if c is not None:
            return c
        if _is_excluded(str(v)):
            return None
    return None
