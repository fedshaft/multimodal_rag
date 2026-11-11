# Reviewing `eval_set.json` — read this before running any evaluation

The 40 questions and their answer keys were **drafted by Claude**, not by a human.
Everything in them was written against real retrieved content — every text chunk
was read, every figure was opened and looked at — and
`eval/validate_eval_set.py` confirms all 62 evidence ids exist and that each
`answer_type` matches the kind of evidence it cites.

**None of that makes the keys correct.** The validator checks *consistency*; whether
a given chunk genuinely answers its question is a judgement only a person can make.
The assignment brief is explicit that LLM-generated data must not be assumed correct,
and a wrong key is worse than a hard question: it scores zero for every configuration
identically, so it looks like a system failure rather than a labelling error, and it
drags down every number in the results table without ever announcing itself.

So this file is the verification pass. Budget about an hour.

**Two of the five flagged items below have now been checked directly: q36 was
correct, q08 was wrong and has been fixed.** One error in the first two checked
is the honest base rate to assume for the rest.

> **Do not give this file, or `eval_set.json`, to the person doing the Cohen's
> Kappa labelling.** Verification and labelling are different jobs and must be
> done by different people, or the agreement score measures nothing. See the
> section at the end.

## What to do

For each question, open the page named in its `verify` field and answer one thing:

> **Does the evidence listed in `relevant_ids` actually answer this question?**

Three outcomes:

- **Fine** — move on.
- **Key is wrong or incomplete** — fix `relevant_ids`. If another chunk answers it
  better, swap it. If two chunks are both needed, list both.
- **Question is bad** — ambiguous, answerable from general knowledge without the
  documents, or answerable from twenty different pages. Rewrite or delete it.
  Deleting is fine; 37 good questions beat 40 shaky ones.

Then re-run:

```
.venv/bin/python eval/validate_eval_set.py
```

## Things I'd check first

These are where a drafted key is most likely to be wrong.

1. ~~**q08 (`unlink`)**~~ — **checked and it was wrong. Now fixed.** The key
   pointed at `file_intro_chunk_30` (p.16), which only *poses* the puzzle
   ("why is this system call named unlink? ... we must first understand
   directories") and defers the answer. The real explanation is in section 39.14
   Hard Links on pp.19–20: a name is a link to an inode, and `unlink()`
   decrements the link count, freeing the data only at zero. Key is now
   `file_intro_chunk_37` + `file_intro_chunk_38`. **This is what a wrong key
   looks like — plausible, on-topic, same document, and completely unanswerable
   from the cited text.**
2. **The `both` questions (q32–q37).** These claim the text *and* the picture are
   each necessary. That is a strong claim. If the prose alone is enough, retag as
   `text` — the by-question-type breakdown is the headline result and it depends
   on this tag being honest.
3. **q29 (file types table)** is a *table*, not a diagram. It is stored as an image
   and text search cannot read it, so `figure` is defensible — but be ready to
   justify it, and consider whether tables deserve their own tag in the write-up.
4. **The page-render ids.** Where a slide figure also has a `_page_` id, both are
   listed, on the grounds that the whole-page render genuinely contains the answer
   too. If you would rather score strictly against the cropped figure, remove the
   `_page_` ids — but do it consistently across all 20 image questions, and say
   which convention you used in the report.
5. ~~**q36 (FCFS disk scheduling)**~~ — **checked, correct.** The slide text on
   ch11 p.19 reads "FCFS — Illustration shows total head movement of 640
   cylinders". SCAN is p.20 and C-SCAN p.22, so the label is right.

## Then: the second annotator

Once the keys are settled, the labels need independent confirmation. This is the
inter-annotator agreement the brief asks for.

```
.venv/bin/python eval/kappa.py --make-sheet eval/annotator_b.csv
```

That writes the 40 questions in **shuffled** order with a blank `label` column
(shuffled deliberately — `eval_set.json` is grouped by type, and handing over a
sheet in that order gives the answer away and makes the agreement score
meaningless). Your teammate labels each one `text` / `figure` / `both` / `none`
**without seeing `eval_set.json`**, then:

```
.venv/bin/python eval/kappa.py --score eval/annotator_b.csv
```

It prints observed agreement, chance agreement, kappa with its Landis & Koch band,
a confusion matrix, and every disagreement itemised.

**Report the kappa from before you resolve disagreements.** That is the number that
measures whether the labelling scheme was clear enough to be applied independently.
Resolving first and reporting after would inflate it and measure nothing. Resolve
afterwards, update `eval_set.json`, and discuss the disagreements in the report —
the disagreements are usually the most interesting paragraph in the section.

## What the set contains

| Type | Count | Share | What it tests |
|---|---:|---:|---|
| `text` | 17 | 42% | Prose answers it — the baseline case |
| `figure` | 14 | 35% | Only an image answers it |
| `both` | 6 | 15% | Needs prose and picture together |
| `none` | 3 | 8% | Not in the collection at all |

20 of 40 (50%) need an image, comfortably above the one-third target.
Evidence is drawn from **30 distinct documents**, so no single chapter dominates.

**Why the figure questions come from the slide decks.** All 14 sit on slides that
carry *no indexed text whatsoever* — their 15–20 words fall below
`MIN_CHUNK_WORDS`, so nothing from those pages is in the text index. BM25 and the
dense index cannot return them at any k. Corpus-wide that is true of **199 of the
371** embedded slide figures. This is the mechanism behind the headline result,
and it is worth stating in the report as a property of the data that was measured
before any model was run — not as something discovered afterwards in the scores.

**The `none` questions** exist because day 4 showed the small vision model will
happily answer from world knowledge rather than admit the documents do not cover
something. Without them that failure is invisible in every metric. All three terms
were confirmed absent from all 69 documents on 2026-09-10.


## Who does what — do not let one person do both

There are two separate jobs here and they **cannot** be done by the same person.

| Job | Who | Sees | Why |
|---|---|---|---|
| **Verify the answer keys** | You | Everything: `eval_set.json`, this file, the PDFs | It is your submission, and you need to know the keys are right |
| **Label for Kappa** | Your teammate | `annotator_b.csv` and the PDFs — **nothing else** | The whole point is an independent judgement |

If your teammate reads the answer keys and then labels the questions, he is
copying, not agreeing, and the kappa is meaningless — it would come out near 1.0
and measure nothing at all. Reporting that number as inter-annotator agreement
would be misrepresenting how the dataset was validated.

So: send him **`eval/annotator_b.csv` and `data/own/`**, and nothing else from
`eval/`. Do your key verification separately.
