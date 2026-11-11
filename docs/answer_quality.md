# Answer-quality evaluation

Generation was evaluated on all 40 questions in `eval/eval_set.json` with the **Full** retrieval
method and **qwen2.5vl:3b** through Ollama (`num_ctx` 8192, temperature from `config.py`), on the
CPU-only development PC (12 cores, 15 GB RAM, no GPU). Run with `python eval/run_generation_eval.py`; raw outputs go to
`eval/results/generation.json` (gitignored, like the other result files).

## How answers were graded

Each answer was compared by hand with the answer key recorded for that question when the
evaluation set was built:

- **correct**: states the key's main point with no wrong claims
- **partly correct**: gets part of the key right, or gives the right method without the specifics asked for
- **wrong**: contradicts the key, misses the point, or says the evidence does not answer it

This is one grader's judgement against fixed keys, not an independent annotation.

## Results

| | Correct | Partly correct | Wrong | Total |
|---|---|---|---|---|
| Text questions | 8 | 1 | 6 | 15 |
| Figure questions | 7 | 3 | 4 | 14 |
| Text + figure questions | 3 | 2 | 3 | 8 |
| **All answerable** | **18 (49%)** | **6 (16%)** | **13 (35%)** | **37** |
| Gold evidence retrieved | 17 | 6 | 7 | 30 |
| Gold evidence missed | 1 | 0 | 6 | 7 |

- **Retrieval drives answer quality.** With the right evidence in the top-k, 17/30 answers are
  correct (23/30 at least partly). Without it, 1/7. Improving retrieval is what moves answers.
- **Refusals on out-of-collection questions:** the automatic matcher (`metrics.abstained`) counts
  1/3. Read by hand, it is 2/3: q40 declines ("not provided in any of the given sources") in
  wording the matcher does not list. q39 (Git rebase) is answered from world knowledge, a real failure.
- **Citations:** precision 0.23, recall 0.28;
  89% of answers contain at least one parseable `[n]` marker. The low
  scores are partly format: the 3B model sometimes drops the markers (q02 is correct with none) or
  cites a neighbouring chunk instead of the gold one.
- **Latency:** 177 s per answer on average on the CPU-only PC; retrieval itself is
  a small fraction of this. The demo Mac runs generation far faster.

## Where answers go wrong

- **Evidence missed** (q08, q14, q18, q19, q23, q35): the model answers from unrelated chunks or
  says the figure does not show it.
- **Evidence retrieved but misread** (q03, q10, q13, q16, q24, q33, q36): typical small-model
  errors, e.g. RAID-1 write bandwidth given as N·R instead of N/2·R, FCFS head order shuffled,
  exponential-averaging formula written wrongly.
- A larger generator (`RAG_GEN_MODEL=qwen2.5vl:7b`) is the obvious next step and needs no code change.

## Per-question results

| Question | Type | Grade | Gold retrieved | Citation recall | Time |
|---|---|---|---|---|---|
| q01 | text | correct | yes | 1.00 | 191 s |
| q02 | text | correct | yes | 0.00 | 188 s |
| q03 | text | wrong | yes | 1.00 | 175 s |
| q04 | text | correct | yes | 0.00 | 184 s |
| q05 | text | correct | yes | 1.00 | 196 s |
| q06 | text | correct | yes | 0.00 | 213 s |
| q07 | text | partly correct | yes | 0.00 | 183 s |
| q08 | text | wrong | no | 0.00 | 195 s |
| q09 | both | correct | yes | 0.50 | 185 s |
| q10 | text | wrong | yes | 0.00 | 177 s |
| q11 | both | partly correct | yes | 0.33 | 186 s |
| q12 | text | correct | yes | 0.50 | 518 s |
| q13 | text | wrong | yes | 0.33 | 175 s |
| q14 | text | wrong | no | 0.00 | 168 s |
| q15 | text | correct | yes | 0.50 | 181 s |
| q16 | text | wrong | yes | 1.00 | 170 s |
| q17 | text | correct | yes | 1.00 | 174 s |
| q18 | figure | wrong | no | 0.00 | 170 s |
| q19 | figure | wrong | no | 0.00 | 184 s |
| q20 | figure | partly correct | yes | 0.50 | 183 s |
| q21 | figure | partly correct | yes | 0.00 | 168 s |
| q22 | figure | correct | yes | 0.50 | 171 s |
| q23 | figure | wrong | no | 0.00 | 129 s |
| q24 | figure | wrong | yes | 0.33 | 108 s |
| q25 | figure | correct | no | 0.00 | 172 s |
| q26 | figure | correct | yes | 0.00 | 164 s |
| q27 | figure | correct | yes | 0.00 | 169 s |
| q28 | figure | correct | yes | 0.00 | 114 s |
| q29 | figure | correct | yes | 0.00 | 117 s |
| q30 | figure | partly correct | yes | 0.00 | 160 s |
| q31 | figure | correct | yes | 0.50 | 170 s |
| q32 | both | correct | yes | 0.67 | 155 s |
| q33 | both | wrong | yes | 0.00 | 118 s |
| q34 | both | correct | yes | 0.33 | 166 s |
| q35 | both | wrong | no | 0.00 | 167 s |
| q36 | both | wrong | yes | 0.00 | 125 s |
| q37 | both | partly correct | yes | 0.50 | 175 s |
| q38 | none | yes (auto), yes (manual) | — | — | refusal expected |
| q39 | none | no (auto), no (manual) | — | — | refusal expected |
| q40 | none | no (auto), yes (manual) | — | — | refusal expected |
