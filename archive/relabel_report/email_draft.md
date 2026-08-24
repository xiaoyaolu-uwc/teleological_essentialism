**To:** David; Shaun
**Subject:** Relabelling done — category shifts + example sentences

Hi David, Hi Shaun,

I've finished relabelling all ~12,900 animal-relevant sentences with the new deployment prompt (91.8% on our evaluation set). Quick update, with a couple of example sentences per category so you can sanity-check the calls. The full report (with a flow diagram of how the labels moved) is attached.

**Headlines**

- 3,903 sentences (30%) changed category. The new prompt is stricter — it only labels a sentence when it makes a positive claim about an actual animal or part — so, as David expected, most of the motion is into junk. Junk rose from 52% to 65%, largely because a lot of the corpus is about plants, humans, God in general, or is index/fragment text.
- Divine teleology dropped 25% and is now a much smaller share: the old prompt fired on almost any mention of God; the new one keeps only sentences that actually assign a purpose to animals.
- The most interesting shift: 428 sentences moved from *internal essence* to *non-divine teleology* — parts formerly read as structural essence are re-read as functional. This is early evidence the structural and functional camps may be partly integrable rather than opposed.

**Example sentences (current labels)**

- *Junk* — Darwin: "Natural selection will never produce in a being anything injurious to itself…" (a statement about the mechanism, not about what an animal is).
- *Divine teleology* — Kirby: "Thus we see how nicely every thing is calculated and adjusted by Supreme Wisdom, to the nature and circumstances of every animal form."
- *Non-divine teleology* — Paley: "MUSCLES, with their tendons, are the instruments by which animal motion is performed."
- *Internal essence* — Haeckel: "Mammals with Cloaca, without Placenta, with Marsupial Bones" (a subclass defined by structural features).

Happy to walk through any of these or go deeper on the internal-essence → non-divine-teleology cell, which I think is the most promising thread.

Best,
xiaoyao
