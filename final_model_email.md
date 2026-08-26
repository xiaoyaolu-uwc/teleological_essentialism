# Final model progress update — draft email

**Subject: The classifier is finished — here's what it can and can't do**

Hi David,

Following up on my last update, where the honest summary was "not yet reliable enough to publish proportions from." That's changed. The model is finished, I've evaluated it properly, and I can now put actual numbers and error bars on what it does. Short version: it's good enough to use, and I can tell you exactly how much to trust it.

---

**The headline.** The model reads a text sentence by sentence, throws out everything that isn't making a real claim about animals, and sorts what remains into our three camps: divine teleology, non-divine teleology, and internal essentialism. Tested on sixteen books it had never seen during training, here's how it does.

*Getting the proportions right.* For a book it has never read, the model's estimate of each category's share lands this close to the truth 90% of the time:

| Category | Within | Typical error |
|---|---|---|
| Divine teleology | ±8.5 points | 2.6 points |
| Internal essentialism | ±15 points | 5.6 points |
| Non-divine teleology | ±16 points | 6.0 points |

The single most encouraging finding is what *isn't* in that table: there's no systematic bias in any category. Every category's average error is within 1.5 points of zero. The model is noisy around the right answer rather than consistently tilted in one direction — which matters enormously for us, because a consistent tilt would quietly bend any trend line we draw over time, whereas noise that averages out won't.

*Getting the ranking right.* This is the better news, and it's the number I'd lean on for the actual argument. Comparing two texts on the same category — which is what every one of our research questions really does — the model gets the ordering right:

| If the true gap between two texts is | We get the ordering right |
|---|---|
| 20 points or more | **99% of the time** |
| 10–20 points | 90% |
| 5–10 points | 84% |
| under 2 points | 54% — i.e. a coin flip |

Read that as a rule of thumb: **a difference of 20 points or more between two periods is solid, 10–20 points is probably real, and anything under 5 points we shouldn't interpret at all.** Within a single book, the model recovers the full ordering of all three categories in 14 of the 16 test books.

The practical upshot is that the model is much better at telling us *which* explanatory style dominates, and how the balance shifts, than at pinning down exact percentages. Given that our claims are comparative — biology versus physics, early versus late, rhetoric versus practice — that's the capability we actually need.

---

**How we got those numbers.** This is worth explaining, because the whole value of the error bars depends on the testing being fair.

The obvious way to test a model is to check it against text it was trained on — but that measures memorisation, not understanding, and it would flatter us badly. What we care about is: hand the model a book it has never seen, how close does it get? So we test it exactly that way. We train the model on some of our books while completely hiding one, then check its answers on the hidden one. Repeat until every book has had a turn being the hidden one. Every number above therefore comes from a book the model had never read a single sentence of.

One wrinkle worth mentioning. The textbook version of this hides exactly one book at a time, which for us would mean training sixteen separate models — expensive, and some of our books are small enough that a single one doesn't give a stable read. So we hide two or three at a time, in six batches, arranged so that no batch is starved of the rarest category (divine teleology is concentrated in only five of the sixteen books, so those five are deliberately spread across different batches). We still get an honest out-of-sample result for all sixteen books, from six rounds of training rather than sixteen.

Then the arithmetic, which is simpler than it sounds. For each of the sixteen books we have two things: the true mix of categories, and the mix the model produced. Subtract one from the other and you get that book's error, per category. Do that for all sixteen and you have sixteen independent readings of how wrong the model tends to be — and the spread of those sixteen numbers *is* the error bar. That's where "±8.5 points" comes from: it's not a theoretical estimate, it's the observed scatter across sixteen real books.

The ranking numbers work the same way. We take every possible pairing of two books, ask whether the model ordered them the same way the truth does, and sort the results by how big the true gap was. That's what produces the table above, and it's why we can say something as specific as "reliable above 20 points."

One statistical note, in the interest of not overselling: those pairwise comparisons aren't fully independent of one another, since sixteen books generate 120 pairs and each book appears in fifteen of them. The standard confidence interval assumes independence and would have given us a tighter range than we've earned. I recomputed it properly by resampling whole books, which widens the interval on that 90% figure to roughly 84–96%. The headline is unchanged, but I'd rather quote the wider, honest number.

---

**What the final model actually is.** Two changes from what I described last time.

The first stage — the relevance filter that was the bottleneck in my last update — is now a small language model called Qwen3-0.6B, fine-tuned for the job, replacing BERT. That was the plan I flagged last time and it worked.

The second change wasn't planned: **I've now replaced the second stage with Qwen as well.** Previously that stage was a version of BERT specially pretrained on historical English, and it was already the strongest part of the pipeline, so I hadn't intended to touch it. But testing the same fine-tuning approach there turned out to roughly halve the error on category proportions and add about eight points of accuracy. Both stages are now the same underlying model, differently fine-tuned — which is also simpler to maintain.

The most consequential fix was mundane. The model reads each sentence in a fixed-size window, and it turned out that window was too small: **28% of our sentences were being silently cut off mid-passage** before the model ever saw the end of them. Widening the window cut the proportion error by roughly a third on its own. That one is slightly embarrassing but worth knowing about, because it had been quietly costing us accuracy for months.

I also tested two further improvements and both came back negative, which I'm recording rather than burying: adjusting how confident the filter has to be before it keeps a sentence made essentially no difference, and averaging several independently trained copies of the filter improved the filter itself but didn't change the end-to-end result. The useful conclusion is that the relevance filter — the thing that was our bottleneck for months — is no longer where the remaining error lives. The two stages now contribute error roughly equally, so further gains would have to come from the categorisation step.

---

**Two caveats I want on the record.**

First, when I say "true proportions," I mean the labels produced by our GPT-5.4 labelling pass, which agrees with our hand-labelled set about 91% of the time. So everything above measures agreement with that labeller, not with ground truth in an absolute sense. The remaining ~9% is a real floor under all these numbers and isn't folded into the error bars.

Second, all sixteen test books are ones we hand-picked as strong representatives of their camps. The broader scanning corpus will be ordinary natural history — more marginal material, a higher share of irrelevant text — so I'd expect these numbers to be somewhat optimistic there. I'll check that on a sample of real corpus texts before we run the full scan.

---

**Where this leaves us.** The classifier is done and its performance is characterised well enough to build on. The next piece is assembling the actual scanning corpus — sampling texts by decade from the Biodiversity Heritage Library — and then running the model over it, which on our current hardware is roughly half a day of compute for a corpus of a couple of thousand texts.

Happy to walk through any of this, and I've written all of it up in the repository if you or anyone else wants the detail.

Best,
Xiaoyao
