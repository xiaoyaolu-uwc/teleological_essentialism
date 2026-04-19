# OCR Cleanup Audit — Reference Texts

> **Date:** 2026-03-28
> **Method:** Random samples at 15%, 50%, 85% of each file + automated artifact counts.

---

## Quality Tiers

Based on the audit, the texts fall into three tiers:

- **Tier 1 (Good OCR, light cleanup needed):** Haeckel Vol 1, Haeckel Vol 2, Lamarck, Mivart, Owen. These are clean 19th-century scans, mostly well-parsed. Minimal artifacts.
- **Tier 2 (Moderate OCR, standard cleanup):** Agassiz, Argyll, Cuvier, Darwin, Gray, Huxley, Kirby Vol 1, Kirby Vol 2, Paley. Readable but with excessive whitespace between words, broken hyphens, page numbers/headers scattered through text.
- **Tier 3 (Rough OCR, aggressive cleanup needed):** Ray, Derham. Both are pre-1800 texts with archaic typography. OCR misreads old typefaces — e.g., long-s ("ſ") sometimes read as "f", ligatures broken, headers/footers badly mangled. Still usable but noisier.

---

## Per-Text Findings

### 1. agassiz_essay_on_classification_1859.txt
- **Size:** 1049KB, 25253 lines
- **Issues:**
  - **Excessive double-spacing:** 131,395 double-space occurrences. Every word is separated by 2+ spaces (OCR artifact from wide-set type).
  - **Broken hyphens:** 3,128 cases of words split across lines with trailing hyphen (e.g., "condi- | tions").
  - **Page numbers:** ~78 standalone number lines scattered through text (page markers).
  - **Index/back matter:** The last ~15% of the file is an index of names — not argumentative prose. Should be stripped.
- **Sample quality:** Prose is readable once spacing is normalized. Good content for Camp A.
- **Cleanup plan:** Collapse multiple spaces → single space. Rejoin hyphenated words across line breaks. Strip lines that are just numbers. Trim index section.

### 2. argyll_reign_of_law_1867.txt
- **Size:** 598KB, 13207 lines
- **Issues:**
  - **Double-spacing:** 81,172 occurrences. Same wide-set type issue as Agassiz.
  - **Broken hyphens:** 1,532 cases.
  - **Semicolons as OCR noise:** Occasional stray semicolons where periods should be (e.g., "motio;ns" for "motions").
- **Sample quality:** Readable. Content includes the hummingbird and orchid chapters we want.
- **Cleanup plan:** Same as Agassiz: collapse spaces, rejoin hyphens. Light character-level fixes.

### 3. cuvier_animal_kingdom_1831.txt
- **Size:** 1088KB, 31412 lines
- **Issues:**
  - **High non-ASCII count:** 911 characters. Mix of Latin species names with accents and OCR misreads.
  - **Broken hyphens:** 2,277.
  - **Page numbers:** 168 standalone number lines.
  - **Taxonomic formatting:** Heavy use of parenthetical references like "(1)" and Latin binomials — these are useful context, not noise. Don't strip.
  - **Very short lines:** Many lines are species entries or footnote references, not continuous prose.
- **Sample quality:** Mixed. The introductory/philosophical sections (first ~10% of text) are well-paragraphed prose. The bulk is taxonomic catalogue entries — short, formulaic descriptions. Both are usable but very different registers.
- **Cleanup plan:** Collapse spaces, rejoin hyphens, strip standalone page numbers. Leave taxonomic formatting intact — it IS the data (Cuvier's functional descriptions of animals).

### 4. darwin_origin_of_species_1859.txt
- **Size:** 1099KB, 22628 lines
- **Issues:**
  - **Double-spacing:** 142,502 occurrences. Heaviest double-spacing of all files.
  - **Broken hyphens:** 1,585.
  - **Page numbers:** 51.
  - **Front matter noise:** First ~2% is publisher/library stamps from the scanned copy.
- **Sample quality:** Excellent once spacing normalized. Classic Darwin prose, highly readable.
- **Cleanup plan:** Collapse spaces, rejoin hyphens, strip page numbers, trim front matter.

### 5. derham_physico_theology_1713.txt
- **Size:** 1159KB, 26605 lines — TIER 3
- **Issues:**
  - **Double-spacing:** 156,543 occurrences. Worst of all files.
  - **Broken hyphens:** 3,818. Highest count.
  - **Page numbers:** 128.
  - **Non-ASCII:** 589. Many are OCR misreads of archaic letterforms.
  - **Archaic spelling:** "firft" for "first", "lefler" for "lesser", "fuch" for "such", "refembhng" for "resembling". This is a mix of genuine long-s usage (ſ→f) and OCR misreads of 18th-century type.
  - **Headers/footers:** Running headers like "Part I. in it/je Cke ATio^. 53" scattered through text — badly OCR'd page headers.
  - **Footnote markers:** Parenthetical numbers like "(5)" mixed into flowing text.
- **Sample quality:** Readable by a human with effort, but noisy. The long-s → "f" substitution is the biggest issue: "first" → "firft", "such" → "fuch", "design" stays "design" (no long-s in that word). This is systematic and partially fixable.
- **Cleanup plan:** Collapse spaces, rejoin hyphens, strip page numbers/headers. For long-s: apply known substitution patterns (firft→first, fuch→such, etc.) but be cautious — some words genuinely had "f" in 18th-century spelling. Consider a dictionary-based approach: try the long-s correction, check if the result is a real English word, keep the correction only if it is.

### 6. gray_darwiniana_1876.txt
- **Size:** 810KB, 16355 lines
- **Issues:**
  - **Double-spacing:** 103,983.
  - **Broken hyphens:** 2,566.
  - **Page numbers:** 17 (relatively few).
- **Sample quality:** Very good. Clean 19th-century American printing, well-parsed by OCR.
- **Cleanup plan:** Standard: collapse spaces, rejoin hyphens.

### 7. haeckel_history_of_creation_vol1_1876.txt
- **Size:** 758KB, 18089 lines — TIER 1
- **Issues:**
  - **Double-spacing:** Only 2! This is essentially clean text.
  - **Broken hyphens:** 1,329 (still present but manageable).
  - **Very few page numbers or non-ASCII artifacts.**
- **Sample quality:** Excellent. Reads like a modern text file. Clean paragraphs, well-formed sentences.
- **Cleanup plan:** Rejoin hyphenated line breaks only. Nearly no other work needed.

### 8. haeckel_history_of_creation_vol2_1876.txt
- **Size:** 714KB, 19493 lines — TIER 1
- **Issues:**
  - **Double-spacing:** Only 3.
  - **Broken hyphens:** 1,146.
  - **Page numbers:** 63 (more than Vol 1, but still minor).
  - **Table/chart noise:** Mid-file has some taxonomic tables with mangled formatting (e.g., "Holocephali Chimeracet chias").
- **Sample quality:** Very good. Some tables will chunk poorly but prose sections are clean.
- **Cleanup plan:** Same as Vol 1. Strip standalone page numbers. Leave taxonomic tables — they'll be filtered out by the animal-keyword step anyway if they're not prose.

### 9. huxley_mans_place_in_nature_1863.txt
- **Size:** 365KB, 8278 lines
- **Issues:**
  - **Double-spacing:** 48,711.
  - **Broken hyphens:** 915.
  - **Page numbers:** 57.
  - **Running headers:** Lines like "FOSSIL REMAINS OF MAN. 165" (chapter title + page number).
- **Sample quality:** Good. Standard Victorian prose, well-OCR'd.
- **Cleanup plan:** Collapse spaces, rejoin hyphens, strip page numbers, strip running headers (regex: lines matching ALL CAPS followed by a number).

### 10. kirby_bridgewater_animals_1835_vol1.txt
- **Size:** 843KB, 21241 lines
- **Issues:**
  - **Double-spacing:** 108,639.
  - **Broken hyphens:** 1,511.
  - **Non-ASCII:** 308. Mix of accented Latin and OCR noise.
  - **Google Books preamble:** First ~500 chars are Google's standard digitization notice ("This is a digital copy of a book that was preserved for ge..."). Should be stripped.
  - **OCR character substitution:** Some garbled characters: "l^s" for "legs", "Georg;e" for "George", "tvill" for "will".
- **Sample quality:** Moderate. Content is excellent (detailed teleological descriptions of insect anatomy and behavior) but OCR quality is spotty.
- **Cleanup plan:** Strip Google preamble. Collapse spaces, rejoin hyphens. Character-level fixes where possible (known OCR substitution patterns). Accept some residual noise.

### 11. kirby_bridgewater_animals_1835_vol2.txt
- **Size:** 792KB, 18485 lines
- **Issues:** Same profile as Vol 1. Double-spacing: 103,285. Broken hyphens: 1,273. Non-ASCII: 340. Google preamble present.
  - **Additional:** "a«" for "as", "kin4" for "kind" — more OCR digit/letter substitution.
- **Cleanup plan:** Same as Vol 1.

### 12. lamarck_zoological_philosophy_1914trans.txt
- **Size:** 1176KB, 27220 lines — TIER 1
- **Issues:**
  - **Double-spacing:** 0! Cleanest file.
  - **Broken hyphens:** 1,115 (only issue).
  - **Page numbers:** 17.
  - **Non-ASCII:** 17 (minimal).
- **Sample quality:** Excellent. Reads like a modern publication. 1914 printing quality + good scan.
- **Cleanup plan:** Rejoin hyphenated line breaks. Strip standalone page numbers. Done.

### 13. mivart_genesis_of_species_1871.txt
- **Size:** 588KB, 15980 lines — TIER 1
- **Issues:**
  - **Double-spacing:** 0.
  - **Broken hyphens:** 1,050.
  - **Page numbers:** 48.
  - **Non-ASCII:** 0! Cleanest non-ASCII count.
  - **Latin quotations:** Some extended Latin passages (Aquinas quotes). These should be preserved — they're part of Mivart's argument.
- **Sample quality:** Excellent.
- **Cleanup plan:** Rejoin hyphens, strip page numbers. Preserve Latin quotations.

### 14. owen_on_the_nature_of_limbs_1849.txt
- **Size:** 258KB, 6296 lines — TIER 1
- **Issues:**
  - **Double-spacing:** 1.
  - **Broken hyphens:** 807.
  - **Page numbers:** 109 (high relative to file size — short text with many page breaks).
  - **Non-ASCII:** 203 (accented names and anatomical Latin terms).
- **Sample quality:** Very good. Clean prose.
- **Cleanup plan:** Rejoin hyphens, strip page numbers. Light touch.

### 15. paley_natural_theology_1802.txt
- **Size:** 801KB, 15992 lines
- **Issues:**
  - **Double-spacing:** 107,718.
  - **Broken hyphens:** 2,341.
  - **Page numbers:** 3 (few).
  - **Front matter:** First ~1% is library accession stamps ("Accessions UA6 Shelf ^so. M£ PJJBLIC, Bosto").
  - **6y for "by":** OCR misreads "b" as "6" in places ("6y the mediation" for "by the mediation").
- **Sample quality:** Good. Classic Paley prose is well-preserved despite spacing issues.
- **Cleanup plan:** Collapse spaces, rejoin hyphens, strip front matter. Fix "6y"→"by" and similar digit-for-letter substitutions.

### 16. ray_wisdom_of_god_1691.txt
- **Size:** 715KB, 16172 lines — TIER 3
- **Issues:**
  - **Double-spacing:** 98,817.
  - **Broken hyphens:** 2,341.
  - **Non-ASCII:** 79.
  - **Archaic typography OCR:** Same long-s issues as Derham but even worse. "firft" for "first", "fuppofe" for "suppose", "fucculent" for "succulent".
  - **Garbled running headers:** "Part I. in it/je Cke ATio^. 53" — page headers badly OCR'd. "Part II. Z;^ /^^ C R E A T I o N. 199" — spaced-out header text.
  - **Mixed letterforms:** ",," for periods, "^" for various characters.
- **Sample quality:** The worst of all files. Readable with effort but noisy. The running headers are the biggest problem — they're scattered through the text every ~25 lines and will pollute chunks.
- **Cleanup plan:** Collapse spaces. Rejoin hyphens. Regex-strip running headers (lines matching "Part [I|II]..." patterns). Apply long-s dictionary correction. Strip front matter (library stamps). Accept ~5–10% residual noise.

---

## Proposed Cleaning Pipeline

All texts get the same base pipeline, then tier-specific extras:

### Base pipeline (all texts):
1. **Strip front matter:** Remove everything before the actual text begins (library stamps, Google notices, publisher marks).
2. **Strip back matter:** Remove indices, appendices, advertising pages.
3. **Collapse whitespace:** Replace 2+ spaces with single space. Normalize line breaks.
4. **Rejoin hyphenated words:** Detect "word- \n" or "word-\n" + next-line-starts-with-lowercase patterns → join into single word.
5. **Strip standalone page numbers:** Remove lines that consist only of a number (with optional whitespace).
6. **Strip running headers:** Remove lines that are ALL CAPS + optional page number, or match known header patterns per text.
7. **Normalize line breaks into paragraphs:** Collapse consecutive non-empty lines into paragraphs separated by double newlines.

### Tier 3 extras (Ray, Derham):
8. **Long-s correction:** Apply dictionary-based "f"→"s" substitution for known 18th-century long-s patterns. Use a word list to validate corrections (only accept if the corrected word exists in an English dictionary).
9. **Aggressive header stripping:** Custom regex per text for the specific garbled header patterns.

### Tier 2 extras (Kirby):
10. **Google preamble removal:** Strip the standard Google Books digitization notice.
11. **Digit-letter OCR fixes:** Known substitutions: "6y"→"by", "l^s"→"legs", "kin4"→"kind", etc. Apply cautiously.

### Output:
Each cleaned file saved as `{original_name}_clean.txt` in a new `clean_texts/` subfolder alongside `raw_texts/`.
