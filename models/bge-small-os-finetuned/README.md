---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:497
- loss:MultipleNegativesRankingLoss
base_model: BAAI/bge-small-en-v1.5
widget:
- source_sentence: 'Represent this sentence for searching relevant passages: RAID
    (0 + 1) and (1 + 0) 11.41 Silberschatz, Galvin and Gagne ©2018 Operating System
    Concepts – 10th Edition'
  sentences:
  - RAID (0 + 1) and (1 + 0) 11.41 Silberschatz, Galvin and Gagne ©2018 Operating
    System Concepts – 10th Edition
  - LRU Approximation Algorithms  LRU needs special hardware and still slow  Reference
    bit • With each page associate a bit, initially = 0 • When page is referenced
    bit set to 1 • Replace any with reference bit = 0 (if one exists) We do not know
    the order, however 10.41 Silberschatz, Galvin and Gagne ©2018 Operating System
    Concepts – 10th Edition
  - Multilevel Queue  Prioritization based upon process type 5.30 Silberschatz, Galvin
    and Gagne ©2018 Operating System Concepts – 10th Edition
- source_sentence: 'Represent this sentence for searching relevant passages: Critical
    Section  General structure of process Pi 6.7 Silberschatz, Galvin and Gagne ©2018
    Operating System Concepts – 10th Edition'
  sentences:
  - Finding parts of a process using process structure c.30 Silberschatz, Galvin and
    Gagne ©2018 Operating System Concepts – 10th Edition
  - 'OpenMP  OpenMP is a set of compiler directives and API that support parallel
    progamming. void update(int value) { #pragma omp critical { count += value } }
     The code contained within the #pragma omp critical directive is treated as a
    critical section and performed atomically. 7.43 Silberschatz, Galvin and Gagne
    ©2018 Operating System Concepts – 10th Edition'
  - Critical Section  General structure of process Pi 6.7 Silberschatz, Galvin and
    Gagne ©2018 Operating System Concepts – 10th Edition
- source_sentence: 'Represent this sentence for searching relevant passages: Multithreaded
    Multicore System  Chip-multithreading (CMT) assigns each core multiple hardware
    threads'
  sentences:
  - Semantics of fork() and exec()  Does fork()duplicate only the calling thread
    or all threads? • Some UNIXes have two versions of fork  exec() usually works
    as normal – replace the running process including all threads 4.50 Silberschatz,
    Galvin and Gagne ©2018 Operating System Concepts – 10th Edition
  - Multithreaded Multicore System  Chip-multithreading (CMT) assigns each core multiple
    hardware threads. (Intel refers to this as hyperthreading.)  On a quad-core system
    with 2 hardware threads per core, the operating system sees 8 logical processors.
    5.41 Silberschatz, Galvin and Gagne ©2018 Operating System Concepts – 10th Edition
  - Dual-Core Design  Multi-chip and multicore  Systems containing all chips • Chassis
    containing multiple separate systems 1.53 Silberschatz, Galvin and Gagne ©2018
    Operating System Concepts – 10th Edition
- source_sentence: 'Represent this sentence for searching relevant passages: General
    Graph Directory (Cont.)  How do we guarantee no cycles'
  sentences:
  - 'OpenMP  OpenMP is a set of compiler directives and API that support parallel
    progamming. void update(int value) { #pragma omp critical { count += value } }
     The code contained within the #pragma omp critical directive is treated as a
    critical section and performed atomically. 7.43 Silberschatz, Galvin and Gagne
    ©2018 Operating System Concepts – 10th Edition'
  - General Graph Directory (Cont.)  How do we guarantee no cycles? • Allow only
    links to files not subdirectories • Garbage collection • Every time a new link
    is added use a cycle detection algorithm to determine whether it is OK 13.33 Silberschatz,
    Galvin and Gagne ©2018 Operating System Concepts – 10th Edition
  - Basic Facts If graph contains no cycles  no deadlock If graph contains a cycle
     • if only one instance per resource type, then deadlock • if several instances
    per resource type, possibility of deadlock 8.11 Silberschatz, Galvin and Gagne
    ©2018 Operating System Concepts – 10th Edition
- source_sentence: 'Represent this sentence for searching relevant passages: Single
    Resource Monitor (Cont.)  Usage: acquire ..'
  sentences:
  - 'Single Resource Monitor (Cont.)  Usage: acquire ... release  Incorrect use
    of monitor operations • release() … acquire() • acquire() … acquire()) • Omitting
    of acquire() and/or release() 6.55 Silberschatz, Galvin and Gagne ©2018 Operating
    System Concepts – 10th Edition'
  - Memory Compression (Cont.)  An alternative to paging is memory compression. 
    Rather than paging out modified frames to swap space, we compress several frames
    into a single frame, enabling the system to reduce memory usage without resorting
    to swapping pages. 10.86 Silberschatz, Galvin and Gagne ©2018 Operating System
    Concepts – 10th Edition
  - 'Dispatch Latency  Conflict phase of dispatch latency: 1. Preemption of any process
    running in kernel mode 2. Release by low- priority process of resources needed
    by high- priority processes 5.49 Silberschatz, Galvin and Gagne ©2018 Operating
    System Concepts – 10th Edition'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on BAAI/bge-small-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). It maps inputs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) <!-- at revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'cls', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Represent this sentence for searching relevant passages: Single Resource Monitor (Cont.) \uf0a7 Usage: acquire ..',
    'Single Resource Monitor (Cont.) \uf0a7 Usage: acquire ... release \uf0a7 Incorrect use of monitor operations • release() … acquire() • acquire() … acquire()) • Omitting of acquire() and/or release() 6.55 Silberschatz, Galvin and Gagne ©2018 Operating System Concepts – 10th Edition',
    'Memory Compression (Cont.) \uf0a7 An alternative to paging is memory compression. \uf0a7 Rather than paging out modified frames to swap space, we compress several frames into a single frame, enabling the system to reduce memory usage without resorting to swapping pages. 10.86 Silberschatz, Galvin and Gagne ©2018 Operating System Concepts – 10th Edition',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8779, 0.4065],
#         [0.8779, 1.0000, 0.4330],
#         [0.4065, 0.4330, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 497 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                           | sentence_2                                                                           |
  |:---------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                             | string                                                                               | string                                                                               |
  | modality | text                                                                               | text                                                                                 | text                                                                                 |
  | details  | <ul><li>min: 17 tokens</li><li>mean: 28.37 tokens</li><li>max: 44 tokens</li></ul> | <ul><li>min: 63 tokens</li><li>mean: 208.73 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 80 tokens</li><li>mean: 222.64 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                     | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | sentence_2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Represent this sentence for searching relevant passages: INTRODUCTION TO OPERATING SYSTEMS 17 2.7 Summary Thus, we have an introduction to the OS</code> | <code>INTRODUCTION TO OPERATING SYSTEMS 17 2.7 Summary Thus, we have an introduction to the OS. Today’s operating systems make systems relatively easy to use, and virtually all operating systems you use today have been inﬂuenced by the developments we will discuss throughout the book. Unfortunately, due to time constraints, there are a number of parts of the OS we won’t cover in the book. For example, there is a lot of net- working code in the operating system; we leave it to you to take the net- working class to learn more about that. Similarly, graphics devices are particularly important; take the graphics course to expand your knowl- edge in that direction. Finally, some operating system books talk a great deal about security; we will do so in the sense that the OS must provide protection between running programs and give users the ability to pro- tect their ﬁles, but we won’t delve into deeper security issues that one might ﬁnd in a security course. However, there are many important topi...</code> | <code>SCHEDULING: INTRODUCTION 11 7.10 Summary We have introduced the basic ideas behind scheduling and developed two families of approaches. The ﬁrst runs the shortest job remaining and thus optimizes turnaround time; the second alternates between all jobs and thus optimizes response time. Both are bad where the other is good, alas, an inherent trade-off common in systems. We have also seen how we might incorporate I/O into the picture, but have still not solved the prob- lem of the fundamental inability of the OS to see into the future. Shortly, we will see how to overcome this problem, by building a scheduler that uses the recent past to predict the future. This scheduler is known as the multi-level feedback queue, and it is the topic of the next chapter. © 2008–23, ARPACI-DUSSEAU THREE EASY PIECES</code>                                                                                                                                                                                                      |
  | <code>Represent this sentence for searching relevant passages: LOCKS 21 References [D91] “Just Win, Baby: Al Davis and His Raiders” by Glenn Dickey</code>     | <code>LOCKS 21 References [D91] “Just Win, Baby: Al Davis and His Raiders” by Glenn Dickey. Harcourt, 1991. The book about Al Davis and his famous quote. Or, we suppose, the book is more about Al Davis and the Raiders, and not so much the quote. To be clear: we are not recommending this book, we just needed a citation. [D+13] “Everything You Always Wanted to Know about Synchronization but Were Afraid to Ask” by Tudor David, Rachid Guerraoui, Vasileios Trigonakis. SOSP ’13, Nemacolin Wood- lands Resort, Pennsylvania, November 2013. An excellent paper comparing many different ways to build locks using hardware primitives. Great to see how many ideas work on modern hardware. [D68] “Cooperating sequential processes” by Edsger W. Dijkstra. 1968. Available online here: http://www.cs.utexas.edu/users/EWD/ewd01xx/EWD123.PDF. One of the early semi- nal papers. Discusses how Dijkstra posed the original concurrency problem, and Dekker’s solution. [H93] “MIPS R4000 Microprocessor User’s Manual” by Joe H...</code> | <code>FLASH-BASED SSDS 21 References [A+08] “Design Tradeoffs for SSD Performance” by N. Agrawal, V. Prabhakaran, T. Wobber, J. D. Davis, M. Manasse, R. Panigrahy. USENIX ’08, San Diego California, June 2008. An excellent overview of what goes into SSD design. [AD14a] “Operating Systems: Three Easy Pieces” by Chapters: Crash Consistency: FSCK and Journaling and Log-Structured File Systems. Remzi Arpaci-Dusseau and Andrea Arpaci-Dusseau. A lot more detail here about how logging can be used in ﬁle systems; some of the same ideas can be ap- plied inside devices too as need be. [AD14a] “Operating Systems: Three Easy Pieces” by Chapters: Locality and the Fast File System and File System Implementation. Remzi Arpaci-Dusseau and Andrea Arpaci-Dusseau. These chapters cover how to build a basic ﬁle system for a hard drive. Amazingly, some of these ideas work perfectly well on SSDs! See if you can ﬁgure out which design techniques are appropriate, and which are less needed. [A15] “Amazon Pricing Study...</code> |
  | <code>Represent this sentence for searching relevant passages: Socket Types  Stream sockets provide reliable, duplex, sequenced data streams</code>           | <code>Socket Types  Stream sockets provide reliable, duplex, sequenced data streams. Supported in Internet domain by the TCP protocol. In UNIX domain, pipes are implemented as a pair of communicating stream sockets.  Sequenced packet sockets provide similar data streams, except that record boundaries are provided • Used in XEROX AF_NS protocol  Datagram sockets transfer messages of variable size in either direction. Supported in Internet domain by UDP protocol.  Reliably delivered message sockets transfer messages that are guaranteed to arrive (Currently unsupported).  Raw sockets allow direct access by processes to the protocols that support the other socket types; e.g., in the Internet domain, it is possible to reach TCP, IP beneath that, or a deeper Ethernet protocol • Useful for developing new protocols c.59 Silberschatz, Galvin and Gagne ©2018 Operating System Concepts – 10th Edition</code>                                                                                                        | <code>STREAMS  STREAM – a full-duplex communication channel between a user-level process and a device in Unix System V and beyond  A STREAM consists of: • STREAM head interfaces with the user process • driver end interfaces with the device • zero or more STREAM modules between them  Each module contains a read queue and a write queue  Message passing is used to communicate between queues • Flow control option to indicate available or busy  Asynchronous internally, synchronous where user process communicates with stream head 12.44 Silberschatz, Galvin and Gagne ©2018 Operating System Concepts – 10th Edition</code>                                                                                                                                                                                                                                                                                                                                                                                                        |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 1
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Time
- **Training**: 29.8 seconds

### Framework Versions
- Python: 3.12.14
- Sentence Transformers: 6.0.1
- Transformers: 5.17.0
- PyTorch: 2.14.0
- Accelerate: 1.15.0
- Datasets: 5.0.1
- Tokenizers: 0.23.2

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->