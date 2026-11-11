# Literature Survey: Local Multimodal RAG for Operating Systems Course Material

## 1. Introduction

Our project answers Operating Systems (OS) course questions from lecture slides and textbook PDFs, including questions whose answer is only in a diagram (for example, a process-state diagram or a page-table figure). It has to run fully on a laptop with no paid APIs.

The system uses **Retrieval-Augmented Generation (RAG)**. RAG means that before a language model answers, a *retriever* first finds the most relevant pieces of the course material, and the model answers using only that evidence. Our pipeline is:

PDF parsing (text chunks, extracted figures, page images) → three indexes (BM25 keyword search, dense text embeddings with BGE-small, CLIP image embeddings) → reciprocal rank fusion → cross-encoder reranker → final selection with reserved image slots and a query-adaptive variant → a local vision-language model (Qwen2.5-VL 3B via Ollama) that writes a cited answer.

This survey reviews **18 peer-reviewed papers published from 2022 to 2025**, grouped into six themes that match the stages of our pipeline. For each paper we note what it did, its key result, and its limitation *for our problem*. We then compare them in one table, identify the research gap, and derive our problem statement and objectives.

**Terms used throughout (defined once):**
- **Chunk**: a short passage of text (a few sentences) cut from a PDF so it can be searched.
- **Sparse / keyword retrieval (BM25)**: ranks passages by how many query words they share, weighted by how rare the words are.
- **Dense retrieval / embedding**: a neural network turns text (or an image) into a list of numbers (a *vector*) so that similar meanings have nearby vectors. Search means finding the nearest vectors.
- **Hybrid retrieval / fusion**: running several retrievers and merging their ranked lists. **Reciprocal Rank Fusion (RRF)** merges lists by giving each item a score of 1/(k + rank) in each list and adding the scores.
- **Reranker / cross-encoder**: a slower, more accurate model that reads the question and one candidate passage *together* and outputs a relevance score. It is used to re-order a short list.
- **Vision-language model (VLM)**: a model that takes images and text as input and generates text.
- **Metrics**: **Recall@5** = fraction of the correct sources found in the top 5; **Hit@5** = whether at least one correct source is in the top 5; **MRR** (mean reciprocal rank) = 1/rank of the first correct result, averaged; **nDCG@10** = a score that rewards putting correct results near the top of the top 10.

**Foundations (older, not counted in the 18):** BM25 (Robertson & Zaragoza, 2009); Reciprocal Rank Fusion (Cormack et al., SIGIR 2009); Dense Passage Retrieval, DPR (Karpukhin et al., EMNLP 2020); the original RAG model (Lewis et al., NeurIPS 2020); CLIP contrastive image-text embeddings (Radford et al., ICML 2021).

---

## 2. Theme 1: RAG foundations and surveys

**[1] Fan et al., "A Survey on RAG Meeting LLMs" (KDD 2024).** This survey organises retrieval-augmented large language models by *what* is retrieved, *how* retrieval and generation are combined, and *how* the system is trained. It is a map of the field rather than an experiment, so it reports no single headline number. **Limitation for us:** it focuses almost entirely on text corpora and large server-side models; diagram-based questions and laptop-only deployment are not its focus.

**[2] Asai et al., "Self-RAG" (ICLR 2024).** Self-RAG trains one language model to decide *when* to retrieve and to output special "reflection tokens" that judge whether a passage is relevant and whether its own answer is supported. The authors report that their 7B and 13B models beat ChatGPT and retrieval-augmented Llama2-chat on open-domain QA, reasoning and fact-verification tasks. **Limitation for us:** it requires fine-tuning the generator with special training data, which is beyond a zero-budget laptop project, and it only retrieves text.

**[3] Ram et al., "In-Context Retrieval-Augmented Language Models" (TACL 2023).** This paper shows that you can get large gains simply by pasting retrieved documents into the prompt of an *unchanged* language model, using off-the-shelf retrievers; adding a reranking step tuned for this setting helps further. **Limitation for us:** it is evaluated on text language modelling only. However, it supports our design choice: we do not retrain the generator, we just put retrieved evidence (including images) into its input.

## 3. Theme 2: Dense, hybrid retrieval and fusion

**[4] Xiao et al., "C-Pack: Packed Resources for General Chinese Embeddings" (SIGIR 2024).** This is the paper behind the BGE embedding models; it releases training data, a benchmark (C-MTEB, 35 datasets, 6 tasks) and models in several sizes, and the authors also release English models that they report as state of the art on the MTEB benchmark. Their Chinese models beat earlier ones by up to +10% on C-MTEB at release time. **Limitation for us:** BGE is a text-only model trained on general web data; it cannot see figures, and in our own experiment further fine-tuning it on synthetic OS pairs *lowered* accuracy, so we kept the base `bge-small-en-v1.5`.

**[5] Bruch et al., "An Analysis of Fusion Functions for Hybrid Retrieval" (ACM TOIS 2023).** The authors compare two ways to merge keyword and dense results: a weighted sum of normalised scores (convex combination) and RRF. They find RRF is *sensitive to its parameters*, and the convex combination beat RRF both in-domain and out-of-domain, needing only a small number of labelled examples to tune. **Limitation for us:** they fuse two *text* retrievers whose scores are comparable; fusing text lists with an image list (CLIP), whose scores behave very differently, is not studied. This motivated our tuning of RRF's `k` and weights.

**[6] Santhanam et al., "ColBERTv2" (NAACL 2022).** ColBERTv2 is a "late-interaction" retriever: it keeps one vector per word of each passage and matches query words to passage words, giving better quality than one-vector models. With compression it reduces storage by 6–10x and reports state-of-the-art results in and out of its training domain (MS MARCO, BEIR). **Limitation for us:** it is still text-only, and the multi-vector index is heavier than a single-vector index on a laptop. Its late-interaction idea later became the basis of ColPali [14].

## 4. Theme 3: Reranking

**[7] Pradeep et al., "Squeezing Water from a Stone" (ECIR 2022).** This work improves a plain BERT-style cross-encoder reranker with a "bag of tricks": an ELECTRA backbone, a better loss with hard negatives (LCE), and a dense first-stage retriever instead of BM25. They reach MRR@10 of 0.414 on the MS MARCO passage development set, nearly 7 points (about 20% relative) above the original monoBERT. **Limitation for us:** it shows that a better first stage plus a cross-encoder is a strong recipe (as in our pipeline), but it is text-only and uses base-size models trained on MS MARCO, not course material.

**[8] Zhuang et al., "RankT5" (SIGIR 2023).** RankT5 adapts the T5 sequence-to-sequence model to output a relevance score directly, and trains it with ranking losses (pairwise or listwise) instead of classification. Listwise training gave better zero-shot results on out-of-domain datasets than classification training. **Limitation for us:** T5 rerankers are larger than our 6-layer MiniLM cross-encoder, which matters for our ~1 s per-query budget on a laptop, and they cannot score images.

**[9] Sun et al., "Is ChatGPT Good at Search?" (EMNLP 2023).** The authors show that well-prompted large language models such as ChatGPT and GPT-4 can rerank passages as well as or better than supervised rerankers, and that a small 440M model distilled from ChatGPT's rankings beats a 3B supervised model on BEIR. **Limitation for us:** the strongest results rely on paid API models, which our zero-budget rule forbids. The distillation result does support using a *small* local reranker.

## 5. Theme 4: Vision-language models and image-text retrieval

**[10] Li et al., "BLIP" (ICML 2022).** BLIP pre-trains one model for both understanding and generation of image-text data, and cleans noisy web captions by generating synthetic captions and filtering bad ones. It reports gains of +2.7% average Recall@1 on image-text retrieval, +2.8% CIDEr on captioning and +1.6% on VQA over prior work. **Limitation for us:** it is trained on natural photos with captions; OS diagrams (boxes, arrows, labels) look very different, so retrieval quality on our figures is not guaranteed.

**[11] Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (SigLIP, ICCV 2023).** SigLIP replaces CLIP's softmax contrastive loss with a simple per-pair sigmoid loss, which does not need to compare every image with every caption in a batch and works better at small batch sizes. A SigLiT model reached 84.5% ImageNet zero-shot accuracy after two days on four TPUv4 chips. **Limitation for us:** like CLIP, it is tuned for photos and short captions; it is a candidate drop-in replacement for our CLIP encoder but was not tested on technical diagrams.

**[12] Liu et al., "Visual Instruction Tuning" (LLaVA, NeurIPS 2023).** LLaVA connects a vision encoder to a language model and trains it on GPT-4-generated image-instruction data, creating a general-purpose visual assistant. It reached an 85.1% relative score versus GPT-4 on a synthetic instruction-following set and 92.53% accuracy on ScienceQA after fine-tuning. **Limitation for us:** it shows that VLMs can answer questions about images, which is the role Qwen2.5-VL plays in our system, but it does not include retrieval: the image must already be given.

## 6. Theme 5: Multimodal and visually rich document RAG

**[13] Ma et al., "Unifying Multimodal Retrieval via Document Screenshot Embedding" (DSE, EMNLP 2024).** DSE skips PDF parsing and embeds a *screenshot* of each page with a large VLM. It beats BM25 by 17 points in top-1 accuracy on Wikipedia pages, and on slide retrieval (SlideVQA-open, 50,714 slides) it reaches nDCG@10 of 75.3 versus 55.8 for BM25 on OCR text. **Limitation for us:** this is the strongest evidence that text-only search misses information in slides, but its encoder is a large VLM, which is expensive to run over a whole course on a laptop.

**[14] Faysse et al., "ColPali" (ICLR 2025).** ColPali embeds page images with a vision-language model and matches them using ColBERT-style late interaction [6]. It also introduces the ViDoRe benchmark. ColPali averaged 81.3 nDCG@5 on ViDoRe versus 67.0 for the best text pipeline (Unstructured parsing + captioning + BGE-M3), and indexed pages in 0.39 s versus 7.22 s for the parsing pipeline (timings reported on an NVIDIA L4 GPU). **Limitation for us:** its benchmark is generic business/scientific documents, not course diagrams, and its timings assume a GPU; we approximate its benefit cheaply by adding a separate CLIP figure index.

**[15] Yu et al., "VisRAG" (ICLR 2025).** VisRAG builds the *whole* RAG pipeline on VLMs: pages are retrieved as images and also passed as images to the generator, avoiding information lost in text parsing. The authors report a 20–40% end-to-end gain over traditional text-based RAG. **Limitation for us:** it uses VLM-based retrievers and generators that are heavier than our laptop budget allows, and it is not evaluated on questions where the answer is a specific diagram inside a lecture deck.

## 7. Theme 6: Evaluating RAG systems

**[16] Es et al., "RAGAs: Automated Evaluation of Retrieval Augmented Generation" (EACL 2024, demo).** RAGAs scores a RAG system without reference answers, using an LLM to measure *faithfulness* (is the answer supported by the retrieved context?), *answer relevance* and *context relevance*. On their WikiEval set, RAGAs agreed with human judgements 95% (faithfulness), 78% (answer relevance) and 70% (context relevance) of the time. **Limitation for us:** it is text-only and by default calls the OpenAI API; it cannot judge whether a diagram was the right evidence.

**[17] Chen et al., "Benchmarking Large Language Models in Retrieval-Augmented Generation" (RGB, AAAI 2024).** RGB tests four abilities: noise robustness, *negative rejection* (refusing when the evidence does not contain the answer), information integration, and counterfactual robustness. Across six LLMs, models handled noise reasonably but struggled significantly with negative rejection, integration and false information. **Limitation for us:** it is text-only and general-knowledge; however, it motivates our three "none" questions, which test whether the system refuses instead of inventing an answer.

**[18] Gao et al., "Enabling Large Language Models to Generate Text with Citations" (ALCE, EMNLP 2023).** ALCE is a benchmark that asks systems to retrieve evidence and write answers with citations, scored for fluency, correctness and citation quality. Even the best models lacked complete citation support about 50% of the time on ELI5. **Limitation for us:** citations point to text passages only; our system must also cite figures and page images.

---

## 8. Master comparison table

| # | Paper | Year | Venue | Method | Dataset | Metrics | Key result | Limitation w.r.t. our problem |
|---|---|---|---|---|---|---|---|---|
| 1 | Fan et al. – RAG meets LLMs survey | 2024 | KDD | Survey / taxonomy of RA-LLMs | (survey) | (qualitative) | Organises retrieval, generation, training choices | Text-centric; no diagram questions or local deployment |
| 2 | Asai et al. – Self-RAG | 2024 | ICLR | LM learns when to retrieve + self-critique tokens | Open-domain QA, reasoning, fact verification | Task accuracy | 7B/13B beat ChatGPT and RA-Llama2-chat | Needs generator fine-tuning; text-only |
| 3 | Ram et al. – In-Context RALM | 2023 | TACL | Paste retrieved docs into frozen LM prompt | Diverse language-modelling corpora | LM performance | Large gains with off-the-shelf retrievers; reranking adds more | Text-only; but supports "no retraining" design |
| 4 | Xiao et al. – C-Pack (BGE) | 2024 | SIGIR | General text embedding models + data + benchmark | C-MTEB (35 datasets), MTEB | Task averages | Up to +10% on C-MTEB; SOTA English MTEB at release | Text-only; general domain (our fine-tuning hurt) |
| 5 | Bruch et al. – Fusion functions | 2023 | ACM TOIS | Convex combination vs RRF for hybrid search | MS MARCO + 8 BEIR sets (in- and out-of-domain) | Recall@1000, NDCG@1000 | Convex combination beats RRF; RRF parameter-sensitive | Only text+text fusion; no image list |
| 6 | Santhanam et al. – ColBERTv2 | 2022 | NAACL | Late-interaction multi-vector retrieval + compression | MS MARCO, BEIR, NQ | Retrieval quality, index size | SOTA quality; 6–10x smaller index | Text-only; heavier index |
| 7 | Pradeep et al. – Cross-encoder tricks | 2022 | ECIR (Springer LNCS) | ELECTRA cross-encoder + LCE loss + dense first stage | MS MARCO passage | MRR@10 | MRR@10 0.414 (dev), ~+7 points over monoBERT | Text-only; general web queries |
| 8 | Zhuang et al. – RankT5 | 2023 | SIGIR | T5 reranker with ranking losses | MS MARCO, NQ; out-of-domain sets (zero-shot) | Ranking quality | Listwise losses generalise better zero-shot | Larger/slower than MiniLM; text-only |
| 9 | Sun et al. – RankGPT | 2023 | EMNLP | LLM permutation reranking + distillation | Standard IR benchmarks incl. BEIR; new NovelEval | Ranking quality | 440M distilled model beats 3B supervised on BEIR | Best results need paid APIs |
| 10 | Li et al. – BLIP | 2022 | ICML | Unified VL pre-training with caption bootstrapping | Image-text retrieval, captioning, VQA benchmarks | Recall@1, CIDEr, VQA score | +2.7% avg R@1 retrieval, +2.8% CIDEr | Natural photos, not technical diagrams |
| 11 | Zhai et al. – SigLIP | 2023 | ICCV | Sigmoid loss for image-text contrastive training | ImageNet (zero-shot) | Zero-shot accuracy | 84.5% ImageNet zero-shot (SigLiT, 4 TPUv4, 2 days) | Photo-oriented; untested on slides |
| 12 | Liu et al. – LLaVA | 2023 | NeurIPS | Vision encoder + LLM, visual instruction tuning | GPT-4-generated instructions, ScienceQA | Relative score, accuracy | 85.1% relative to GPT-4; 92.53% ScienceQA | No retrieval; image must be given |
| 13 | Ma et al. – DSE | 2024 | EMNLP | Embed page screenshots with a VLM | Wiki-SS (1.3M pages), SlideVQA-open | Top-1 acc., nDCG@10 | Slides nDCG@10 75.3 vs 55.8 (BM25 on OCR) | Large VLM encoder; costly on laptop |
| 14 | Faysse et al. – ColPali | 2025 | ICLR | VLM page embeddings + late interaction | ViDoRe | nDCG@5, indexing latency | 81.3 vs 67.0 nDCG@5; 0.39 s vs 7.22 s per page | Generic docs; GPU timings |
| 15 | Yu et al. – VisRAG | 2025 | ICLR | Retrieve and generate on page images with VLMs | Multi-modality document QA sets | Retrieval & end-to-end accuracy | 20–40% end-to-end gain over text RAG | Heavy VLM retriever + generator |
| 16 | Es et al. – RAGAs | 2024 | EACL (demo) | Reference-free LLM-judged RAG metrics | WikiEval (50 Wikipedia pages) | Agreement with humans | 0.95 / 0.78 / 0.70 agreement (faith. / ans. / ctx.) | Text-only; default uses paid API |
| 17 | Chen et al. – RGB | 2024 | AAAI | Benchmark of 4 RAG abilities | RGB (English + Chinese) | Scores per ability | LLMs weak at negative rejection & integration | Text-only; general knowledge |
| 18 | Gao et al. – ALCE | 2023 | EMNLP | Benchmark for answers with citations | ASQA, QAMPARI, ELI5 | Fluency, correctness, citation quality | Best models lack full citation support ~50% (ELI5) | Citations to text only, not figures |

---

## 9. Critical comparison

**Methods.** The papers split into two families. *Text-first* systems [2–9, 16–18] parse PDFs into text and use keyword search, dense embeddings, fusion and rerankers. They are cheap and mature, but anything that exists only as a picture is invisible to them. *Vision-first* systems [13–15] skip parsing and embed whole page images with large VLMs; they recover visual information but need models of roughly the same size as the generator, plus a GPU, for every page. General vision-language models [10–12] supply the building blocks (image encoders, VLM generators) but are trained on photos and do not retrieve.

**Datasets.** Almost every retrieval and reranking result comes from general web or Wikipedia text (MS MARCO, BEIR, NQ) [4–9]. Visually rich benchmarks (ViDoRe [14], SlideVQA-open in [13], VisRAG's sets [15]) use business reports, scientific papers or generic slide decks. None of the 18 papers evaluate on *course material*, where the same concept appears in both a textbook paragraph and a slide diagram, and where a student may explicitly ask for "the diagram".

**Results.** Two consistent findings matter for us. (1) Combining a good first stage with a cross-encoder reranker reliably improves ranking [3, 7, 9]; our own reranker raised MRR from 0.475 to 0.556. (2) When the evidence is visual, text-only retrieval loses a lot: DSE beat OCR-text BM25 by about 20 nDCG@10 points on slides [13], and ColPali beat the best text pipeline by about 14 nDCG@5 points [14]. Bruch et al. [5] add a warning that RRF is sensitive to its settings, which matches our experience that adding the CLIP list changed the balance between text and figure results.

**Evaluation.** Existing RAG evaluation tools [16–18] judge text answers and text citations. They do not report results separately for "answer is in text" versus "answer is in a figure", so they can hide a complete failure on one question type behind a good average.

## 10. Research gap

Putting the papers together:

1. **Text-first RAG is blind to diagrams, and the literature mostly measures it on text-only benchmarks.** Our own measurements confirm this sharply. On the 14 figure-only questions in our 40-question set, **BM25, dense (BGE) and hybrid retrieval all score 0.0 Recall@5, 0.0 Hit@5 and 0.0 MRR**: not one correct figure is retrieved. Their overall averages (Recall@5 0.36–0.41) look moderate only because they do well on text questions (hybrid: 0.856 Recall@5 on text questions).
2. **The methods that do handle visual pages [13–15] are built for GPU servers.** They use multi-billion-parameter VLMs as retrievers and are benchmarked on generic documents. There is little evidence on what a *laptop-only, zero-budget* system can achieve on lecture slides plus textbook figures.
3. **Fusing text and image result lists is under-studied.** Fusion studies [5] combine two text retrievers. In our system, adding the CLIP image list raised figure-question Recall@5 from 0.0 to 0.417, but text-question Recall@5 dropped from 0.856 to 0.589, because images pushed correct text out of the top 5. The text-only reranker recovered text recall to 0.756. How to fuse and select across modalities without one crowding out the other (our reserved image slots and query-adaptive selection) is not addressed by the surveyed papers.
4. **Evaluation does not break results down by modality.** RAG benchmarks [16–18] and retrieval benchmarks [4–9] report single averages on text. A small, reliably labelled test set that separates text, figure, both and unanswerable questions, for a specific course, does not exist in the reviewed work.
5. **Domain adaptation is not free.** Embedding and reranking papers [4, 7, 8] train on large general datasets. In our setting, fine-tuning the embedder and reranker on synthetic OS question pairs *reduced* accuracy (negative transfer), so general pre-trained models remained the better choice for a small course corpus.

**Gap in one sentence:** there is no evaluated, laptop-only RAG system for course material that retrieves *both* text passages and diagrams, fuses them without one crowding out the other, and reports performance separately for figure questions, where text-only methods in our tests score 0.0.

## 11. Problem statement

Students revising Operating Systems need answers that may live in textbook paragraphs or only in lecture-slide diagrams. Existing text-based RAG pipelines cannot retrieve diagram-only evidence (0.0 Recall@5 on our figure questions), while page-image retrievers that can are designed for GPU servers. **The problem is to design and evaluate a fully local, zero-cost multimodal RAG system that retrieves text chunks and figures from OS course PDFs, merges and reranks them so that both modalities reach the final evidence set, and generates answers that cite their sources, with performance measured separately by question type on a reliably labelled test set.**

## 12. Objectives

Each objective has a measurable target. Baselines and results are from our 40-question evaluation (37 questions with a labelled source: 15 text, 14 figure, 8 both; 3 "none" questions). The "Measured" column shows what our system achieved.

| # | Objective | Metric and target | Baseline | Measured |
|---|---|---|---|---|
| O1 | Retrieve diagram evidence that text-only search misses | Figure-question Recall@5 from 0.0 to **≥ 0.30** | 0.0 (BM25, dense, hybrid) | 0.417 (full system); 0.345 (tuned adaptive) |
| O2 | Improve overall retrieval over the best text-only hybrid | Overall Recall@5 **≥ 0.50** and Hit@5 **≥ 0.75** | Hybrid 0.41 / 0.514 | Full 0.554 / 0.811 |
| O3 | Improve ranking quality with a local cross-encoder reranker without losing text questions | Overall MRR **≥ 0.50**; text-question Recall@5 within **0.10** of the hybrid's 0.856 | +CLIP without reranker: MRR 0.475, text R@5 0.589 | Full MRR 0.556, text R@5 0.756; tuned MRR 0.568, text R@5 0.811 |
| O4 | Run entirely on a laptop at zero cost | Retrieval + reranking latency **≤ 1.5 s per query**; **0** paid API calls | — | ≈ 1.0 s per query (1007 ms full, 997 ms tuned); all models local (Ollama) |
| O5 | Build a reliable, modality-labelled test set and report by question type | **40** questions across text / figure / both / none; inter-annotator **Cohen's kappa ≥ 0.70** (kappa measures agreement between two labellers beyond chance) | — | 40 questions; kappa = 0.75; results reported per type |

Note on the trade-off: the tuned adaptive variant gives the best ranking (MRR 0.568, nDCG@10 0.495) and best text-question recall, while the full system gives the best figure-question recall (0.417). Which to deploy depends on whether figure questions or ranking quality matter more.

---

## References

[1] W. Fan, Y. Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and Q. Li, "A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models," in *Proc. 30th ACM SIGKDD Conf. Knowledge Discovery and Data Mining (KDD)*, 2024, pp. 6491–6501. https://doi.org/10.1145/3637528.3671470

[2] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html

[3] O. Ram, Y. Levine, I. Dalmedigos, D. Muhlgay, A. Shashua, K. Leyton-Brown, and Y. Shoham, "In-Context Retrieval-Augmented Language Models," *Trans. Assoc. Comput. Linguistics (TACL)*, vol. 11, pp. 1316–1331, 2023. https://aclanthology.org/2023.tacl-1.75/

[4] S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, and J.-Y. Nie, "C-Pack: Packed Resources For General Chinese Embeddings," in *Proc. 47th Int. ACM SIGIR Conf. Research and Development in Information Retrieval*, 2024, pp. 641–649. https://doi.org/10.1145/3626772.3657878

[5] S. Bruch, S. Gai, and A. Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval," *ACM Trans. Information Systems (TOIS)*, vol. 42, no. 1, 2023. https://doi.org/10.1145/3596512

[6] K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia, "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction," in *Proc. NAACL-HLT*, 2022, pp. 3715–3734. https://aclanthology.org/2022.naacl-main.272/

[7] R. Pradeep, Y. Liu, X. Zhang, Y. Li, A. Yates, and J. Lin, "Squeezing Water from a Stone: A Bag of Tricks for Further Improving Cross-Encoder Effectiveness for Reranking," in *Advances in Information Retrieval (ECIR 2022)*, LNCS vol. 13185, Springer, 2022, pp. 655–670. https://doi.org/10.1007/978-3-030-99736-6_44

[8] H. Zhuang, Z. Qin, R. Jagerman, K. Hui, J. Ma, J. Lu, J. Ni, X. Wang, and M. Bendersky, "RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses," in *Proc. 46th Int. ACM SIGIR Conf.*, 2023, pp. 2308–2313. https://doi.org/10.1145/3539618.3592047

[9] W. Sun et al., "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2023, pp. 14918–14937. https://aclanthology.org/2023.emnlp-main.923/

[10] J. Li, D. Li, C. Xiong, and S. Hoi, "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation," in *Proc. 39th Int. Conf. Machine Learning (ICML)*, PMLR vol. 162, 2022, pp. 12888–12900. https://proceedings.mlr.press/v162/li22n.html

[11] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, "Sigmoid Loss for Language Image Pre-Training," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2023, pp. 11975–11986. https://doi.org/10.1109/ICCV51070.2023.01100

[12] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html

[13] X. Ma, S.-C. Lin, M. Li, W. Chen, and J. Lin, "Unifying Multimodal Retrieval via Document Screenshot Embedding," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2024, pp. 6492–6505. https://aclanthology.org/2024.emnlp-main.373/

[14] M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud, C. Hudelot, and P. Colombo, "ColPali: Efficient Document Retrieval with Vision Language Models," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/99e9e141aafc314f76b0ca3dd66898b3-Abstract-Conference.html

[15] S. Yu, C. Tang, B. Xu, J. Cui, J. Ran, Y. Yan, Z. Liu, S. Wang, X. Han, Z. Liu, and M. Sun, "VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/3640a1997a4c9571cea9db2c82e1fc35-Abstract-Conference.html

[16] S. Es, J. James, L. Espinosa Anke, and S. Schockaert, "RAGAs: Automated Evaluation of Retrieval Augmented Generation," in *Proc. 18th Conf. European Chapter of the ACL (EACL): System Demonstrations*, 2024, pp. 150–158. https://aclanthology.org/2024.eacl-demo.16/

[17] J. Chen, H. Lin, X. Han, and L. Sun, "Benchmarking Large Language Models in Retrieval-Augmented Generation," in *Proc. AAAI Conf. Artificial Intelligence*, vol. 38, no. 16, 2024, pp. 17754–17762. https://doi.org/10.1609/aaai.v38i16.29728

[18] T. Gao, H. Yen, J. Yu, and D. Chen, "Enabling Large Language Models to Generate Text with Citations," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2023, pp. 6465–6488. https://aclanthology.org/2023.emnlp-main.398/
