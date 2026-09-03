#!/usr/bin/env python3
"""Split scanned rows by whether they discuss an animal PART or a whole animal.

config.ANIMAL_ANATOMY is not usable directly: it mixes true parts (wing, beak,
vertebra) with behaviours and life-stages (instinct, migration, hibernation,
nest, carnivore). Those behaviour terms are exactly what we suspect inflates
NDT, so counting them as "parts" would defeat the test.

The vocabulary below is parts only. It is deliberately a plain word-set match,
not a regex over the sentence text, so a term matches as a whole word.
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SENT = ROOT / "data/scan/sentences"
PRED = ROOT / "data/scan/predictions"

PARTS = """
wing wings fin fins claw claws talon talons beak beaks bill bills feather feathers
fur hide scale scales limb limbs leg legs foot feet paw paws hoof hooves hoofs
horn horns tail tails antler antlers tusk tusks vertebra vertebrae skeleton skull
skulls bone bones jaw jaws mandible mandibles tooth teeth tongue tongues eye eyes
ear ears nostril nostrils snout muzzle beard crest comb wattle
heart lung lungs liver kidney kidneys stomach stomachs intestine intestines gut
gill gills lungfish trachea gullet oesophagus esophagus
antenna antennae palp palpi proboscis mouthparts thorax abdomen abdomens elytra
elytron carapace shell shells plastron mantle siphon tentacle tentacles
muscle muscles nerve nerves brain brains spine spinal rib ribs sternum pelvis
femur tibia tarsus metatarsus humerus radius ulna clavicle scapula
gland glands membrane membranes cartilage ligament tendon
hair hairs bristle bristles spine spines quill quills plume plumes
udder teat teats hump snail_foot digit digits toe toes nail nails hooflet
""".split()
PARTS = {w for w in PARTS if "_" not in w}

TOKEN = re.compile(r"[A-Za-z]+")


def has_part(text: str) -> bool:
    return any(t.lower() in PARTS for t in TOKEN.findall(text))


def main() -> None:
    meta = {w["work_id"]: w["subfield"]
            for w in csv.DictReader((ROOT / "corpus/bhl/derived/scan_pool.csv").open())}
    # uid -> work_id, from the sentence files themselves
    cats = ["divine_teleology", "non_divine_teleology", "internal_essence"]
    agg = defaultdict(lambda: defaultdict(int))

    for pred_path in sorted(PRED.glob("*.csv")):
        sent_path = SENT / pred_path.name
        if not sent_path.exists() or pred_path.stat().st_size == 0:
            continue
        srows = list(csv.DictReader(sent_path.open()))
        prows = list(csv.DictReader(pred_path.open()))
        if len(srows) != len(prows):
            continue
        for s, p in zip(srows, prows):
            if p["gate_pred"] == "junk" or p["s2_pred"] not in cats:
                continue
            bucket = "part" if has_part(s["text"]) else "whole"
            key = (s["period_bin"], meta.get(s["work_id"], "(IA supplement)"), bucket)
            agg[key]["n"] += 1
            agg[key][p["s2_pred"]] += 1

    out = []
    for (period, subfield, bucket), c in sorted(agg.items()):
        out.append({"period_bin": period, "subfield": subfield, "bucket": bucket,
                    "n_explanatory": c["n"],
                    **{f"n_{k}": c[k] for k in cats}})
    dest = ROOT / "data/scan/scan_by_parts.csv"
    with dest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    print(f"{'bucket':<8}{'rows':>10}{'DT':>7}{'NDT':>7}{'IE':>7}")
    for bucket in ("whole", "part"):
        rows = [r for r in out if r["bucket"] == bucket]
        t = sum(r["n_explanatory"] for r in rows)
        vals = [100 * sum(r[f"n_{c}"] for r in rows) / t for c in cats]
        print(f"{bucket:<8}{t:>10,}" + "".join(f"{v:>7.1f}" for v in vals))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
