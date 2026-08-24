# BERT progress update — draft email

**Subject: BERT progress update — where things stand and what's next**

Hi [David / PI names],

It's been a while since I last checked in on the BERT side of the project, so I wanted to give you a real update on where things stand, what we've learned, and what I'm doing next.

**The goal.** The reason we're training BERT at all is so we can eventually point it at large volumes of historical text and get a trustworthy read on how the balance between two explanatory styles — divine/non-divine teleology (animals explained by purpose) versus internal essentialism (animals explained by structure) — shifted over time. For that to be useful, the single most important thing the model has to get right isn't whether it nails every individual sentence; it's whether, across a whole text, the *overall proportions* of these categories come out close to correct. We can tolerate the model being overly cautious and throwing out some borderline material as irrelevant ("junk"), as long as that over-caution doesn't disproportionately eat into one category more than the others — and when the model does commit to a real category, we need it to actually be right.

**Where we are.** Our current approach uses BERT to first sort each passage into "relevant" or "not relevant," and then, for everything judged relevant, a second, more specialized pass decides which of the three explanatory camps it belongs to. We test the model the way we'll actually use it in the future: by training it on a set of texts while completely holding out one book it's never seen, then checking how well it does on that unseen book — a fair proxy for pointing it at brand-new material later.

The honest summary is: the second step — sorting relevant material into the right camp — is genuinely good, correctly categorizing 85–90% of passages once given real content, which is competitive with our best earlier approach (an LLM prompt we'd already refined and used to label the full corpus). The bottleneck is the first step, the relevance filter: it wrongly discards a meaningful share of real content as irrelevant, and it does so unevenly across categories rather than uniformly. Because of that, the two steps chained together — which is what an actual deployed model would look like — only perform moderately well overall, even though the second step alone is strong.

**On the metric that matters most to us.** We looked specifically at whether the model's output proportions match the true proportions for a given text, on the two test books where we could check this directly:

| Text | | Divine teleology | Non-divine teleology | Internal essentialism |
|---|---|---|---|---|
| **Reign of Law** | True proportion | 17% | 67% | 16% |
| | Our current pipeline's output | 20% | 63% | 17% |
| | Output *if the relevance filter were perfect* | 22% | 62% | 16% |
| **Darwiniana** | True proportion | 15% | 55% | 30% |
| | Our current pipeline's output | 21% | 63% | 16% |
| | Output *if the relevance filter were perfect* | 16% | 47% | 37% |

Each block has three rows: the true proportion for that book (our best available answer, from the earlier LLM-based approach); what our current two-step pipeline actually produces; and what the categorization step alone would produce if we swapped in a hypothetically perfect relevance filter, isolating how much of the error is coming from that first step specifically. On Reign of Law, our current pipeline is reasonably close to true across all three categories (within a few percentage points), and a better filter wouldn't change much. On Darwiniana, however, the pipeline meaningfully overstates divine teleology's share and, more strikingly, understates internal essentialism's share by close to half (30% true vs. 16% observed) — and a perfect filter would actually flip that error, *overstating* internal essentialism instead (37%). That flip is the clearest evidence that the categorization step itself is basically trustworthy, but the relevance filter's mistakes are inconsistent from book to book in a way that isn't yet predictable or correctable, which is exactly the risk we can't accept for a real proportion-over-time claim.

We also confirmed that when real content does get wrongly discarded as irrelevant, it isn't spread evenly across the three categories — meaning any proportions we'd report right now could be quietly skewed in a way we can't yet predict. The reverse problem — irrelevant material sneaking through and getting mislabeled as one of the three real categories — turned out to be small (under 10% of true irrelevant material) and fairly evenly spread, so that's not where the risk is. As a side note, we also tested whether using a version of BERT specifically pretrained on historical-style English (rather than a generic modern one) helped with this relevance-sorting step, and surprisingly it didn't consistently — which tells us the problem isn't really about the model struggling with old-fashioned language, it's something else about how it's drawing that particular line.

**Bottom line:** the model is not yet reliable enough to publish proportion estimates from. The good news is we now know precisely where the problem lives (the relevance filter, not the categorization itself) and that the categorization step is already close to production quality.

**Next steps.** We're committing to fine-tuning a small language model specifically for the relevance-filtering step, using a lightweight fine-tuning method called LoRA — this is a bigger architectural change than tweaking the current BERT filter, and we think it's the right next investment given the filter is the clear bottleneck rather than a minor knob to turn. In parallel, we're going to build a formal way of expressing uncertainty around any proportion number we eventually report: rather than trying to mathematically "correct" the model's raw output (which carries its own risks), we'll compute a margin of error for each proportion estimate based on the model's known track record on texts where we already have trustworthy answers — so any claim we make will come with an honest, defensible range rather than a bare percentage.

Happy to talk through any of this in more detail whenever works.

Best,
Xiaoyao
