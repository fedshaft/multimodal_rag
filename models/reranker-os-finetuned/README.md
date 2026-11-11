---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:540
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision 233902d25c440f23af6f7d6e94d2946bac0bee0a -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
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
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['an approach which tries to segregate hot and cold seg- ments', 'an approach which tries to segregate hot and cold seg- ments. A hot segment is one in which the contents are being frequently over-written; thus, for such a segment, the best policy is to wait a long time before cleaning it, as more and more blocks are getting over-written (in new segments) and thus being freed for use. A cold segment, in con- trast, may have a few dead blocks but the rest of its contents are relatively stable. Thus, the authors conclude that one should clean cold segments sooner and hot segments later, and develop a heuristic that does exactly that. However, as with most policies, this policy isn’t perfect; later ap- proaches show how to do better [MR+97]. © 2008–23, ARPACI-DUSSEAU THREE EASY PIECES'],
    ['8 LOG-STRUCTURED FILE SYSTEMS 43.8 What About Directories', '8 LOG-STRUCTURED FILE SYSTEMS 43.8 What About Directories? Thus far, we’ve simpliﬁed our discussion a bit by only considering in- odes and data blocks. However, to access a ﬁle in a ﬁle system (such as /home/remzi/foo, one of our favorite fake ﬁle names), some directo- ries must be accessed too. So how does LFS store directory data? Fortunately, directory structure is basically identical to classic UNIX ﬁle systems, in that a directory is just a collection of (name, inode number) mappings. For example, when creating a ﬁle on disk, LFS must both write a new inode, some data, as well as the directory data and its inode that refer to this ﬁle. Remember that LFS will do so sequentially on the disk (after buffering the updates for some time). Thus, creating a ﬁle foo in a directory would lead to the following new structures on disk: I[k] b[0]:A0 (foo, k) I[dir] b[0]:A2 m[k]:A1 m[dir]:A3 Dk Ddir imap A0 A1 A2 A3 The piece of the inode map contains the information for the location of both the directory ﬁle dir as well as the newly-created ﬁle f. Thus, when accessing ﬁle foo (with inode number k), you would ﬁrst look in the inode map (usually cached in memory) to ﬁnd the location of the inode of directory dir (A3); you then read the directory inode, which gives you the location of the directory data (A2); reading this data block gives you the name-to-inode-number mapping of (foo, k). You then consult the inode map again to ﬁnd the location of inode number k (A1), and ﬁnally read the desired data block at address A0. There is one other serious problem in LFS that the inode map solves, known as the recursive update problem [Z+12]. The problem arises in any ﬁle system that'],
    ['Kernel — Scheduling \uf0a7 The dispatcher uses a 32-level priority scheme to determine the order of thread execution', 'Windows Scheduling \uf0a7 Windows uses priority-based preemptive scheduling \uf0a7 Highest-priority thread runs next \uf0a7 Dispatcher is scheduler \uf0a7 Thread runs until (1) blocks, (2) uses time slice, (3) preempted by higher-priority thread \uf0a7 Real-time threads can preempt non-real-time \uf0a7 32-level priority scheme \uf0a7 Variable class is 1-15, real-time class is 16-31 \uf0a7 Priority 0 is memory-management thread \uf0a7 Queue for each priority \uf0a7 If no run-able thread, runs idle thread 5.65 Silberschatz, Galvin and Gagne ©2018 Operating System Concepts – 10th Edition'],
    ['make sure that only proper forms of access to it are allowed', 'make sure that only proper forms of access to it are allowed. So merely relying on vir- tualization to ensure proper access just pushes the problem down to pro- tecting the virtualization functionality of the OS. Even if we leave that issue aside, sooner or later we have to move past cheap special cases and deal with the general problem. Subject X wants to read and write object /tmp/foo. Maybe it’s allowable, maybe it isn’t. Now what? Computer scientists have come up with two basic approaches to solv- ing this question, relying on different data structures and different meth- ods of making the decision. One is called access control lists and the other is called capabilities. It’s actually a little inaccurate to claim that computer scientists came up with these approaches, since they’ve been in use in non-computer contexts for millennia. Let’s look at them in a more general perspective before we consider operating system implementa- tions. Let’s say we want to start an exclusive nightclub (called, perhaps, Chez Andrea3) restricted to only the best operating system researchers and developers. We don’t want to let any of those database or program- ming language people slip in, so we’ll need to make sure only our ap- proved customers get through the door. How might we do that? One 2Almost. Remember the bits in the page table that determine whether a particular page can be read, written, or executed? But it’s not the operating system doing the runtime check here, it’s the virtual memory hardware. 3The authors Arpaci-Dusseau would like to note that author Reiher is in charge of these name choices for the security chapters, and did not strong-arm him into using their names throughout this and other examples. We now return you to your regular reading... © 2008–23, ARPACI-DUSSEAU (OSTEP)'],
    ['12 DISTRIBUTED SYSTEM SECURITY will continue to assume that the authentication is valid', '12 DISTRIBUTED SYSTEM SECURITY will continue to assume that the authentication is valid. It won’t ask for your password every time you click a link or perform some other inter- action with it. (And a good thing, too. Imagine how much of a pain it would be if you had to provide your password every time you wanted to do anything.) If your session is encrypted at this point, it could regard your proper use of the cryptography as a form of authentication; but you might even be able to quit your web browser, start it up again, navigate back to that web site, and still be treated as an authenticated user, with- out a new request for your password. At that point, you’re no longer using the same cryptography you used before, since you would have es- tablished a new session and set up a new cryptographic key. How did your partner authenticate that you were the one receiving the new key? In such cases, the site you are working with has chosen to make a se- curity tradeoff. It veriﬁed your identity at some time in the past using your password and then relies on another method to authenticate you in the future. A common method is to use web cookies. Web cookies are pieces of data that a web site sends to a client with the intention that the client stores that data and send it back again whenever the client next communicates with the server. Web cookies are built into most browsers and are handled invisibly, without any user intervention. With proper use of cryptography, a server that has veriﬁed the password of a client can create a web cookie that securely stores the client’s identity. When the client communicates with the server again, the web'],
]
scores = model.predict(pairs)
print(scores)
# [9.5255 7.3615 0.495  7.922  9.85  ]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'an approach which tries to segregate hot and cold seg- ments',
    [
        'an approach which tries to segregate hot and cold seg- ments. A hot segment is one in which the contents are being frequently over-written; thus, for such a segment, the best policy is to wait a long time before cleaning it, as more and more blocks are getting over-written (in new segments) and thus being freed for use. A cold segment, in con- trast, may have a few dead blocks but the rest of its contents are relatively stable. Thus, the authors conclude that one should clean cold segments sooner and hot segments later, and develop a heuristic that does exactly that. However, as with most policies, this policy isn’t perfect; later ap- proaches show how to do better [MR+97]. © 2008–23, ARPACI-DUSSEAU THREE EASY PIECES',
        '8 LOG-STRUCTURED FILE SYSTEMS 43.8 What About Directories? Thus far, we’ve simpliﬁed our discussion a bit by only considering in- odes and data blocks. However, to access a ﬁle in a ﬁle system (such as /home/remzi/foo, one of our favorite fake ﬁle names), some directo- ries must be accessed too. So how does LFS store directory data? Fortunately, directory structure is basically identical to classic UNIX ﬁle systems, in that a directory is just a collection of (name, inode number) mappings. For example, when creating a ﬁle on disk, LFS must both write a new inode, some data, as well as the directory data and its inode that refer to this ﬁle. Remember that LFS will do so sequentially on the disk (after buffering the updates for some time). Thus, creating a ﬁle foo in a directory would lead to the following new structures on disk: I[k] b[0]:A0 (foo, k) I[dir] b[0]:A2 m[k]:A1 m[dir]:A3 Dk Ddir imap A0 A1 A2 A3 The piece of the inode map contains the information for the location of both the directory ﬁle dir as well as the newly-created ﬁle f. Thus, when accessing ﬁle foo (with inode number k), you would ﬁrst look in the inode map (usually cached in memory) to ﬁnd the location of the inode of directory dir (A3); you then read the directory inode, which gives you the location of the directory data (A2); reading this data block gives you the name-to-inode-number mapping of (foo, k). You then consult the inode map again to ﬁnd the location of inode number k (A1), and ﬁnally read the desired data block at address A0. There is one other serious problem in LFS that the inode map solves, known as the recursive update problem [Z+12]. The problem arises in any ﬁle system that',
        'Windows Scheduling \uf0a7 Windows uses priority-based preemptive scheduling \uf0a7 Highest-priority thread runs next \uf0a7 Dispatcher is scheduler \uf0a7 Thread runs until (1) blocks, (2) uses time slice, (3) preempted by higher-priority thread \uf0a7 Real-time threads can preempt non-real-time \uf0a7 32-level priority scheme \uf0a7 Variable class is 1-15, real-time class is 16-31 \uf0a7 Priority 0 is memory-management thread \uf0a7 Queue for each priority \uf0a7 If no run-able thread, runs idle thread 5.65 Silberschatz, Galvin and Gagne ©2018 Operating System Concepts – 10th Edition',
        'make sure that only proper forms of access to it are allowed. So merely relying on vir- tualization to ensure proper access just pushes the problem down to pro- tecting the virtualization functionality of the OS. Even if we leave that issue aside, sooner or later we have to move past cheap special cases and deal with the general problem. Subject X wants to read and write object /tmp/foo. Maybe it’s allowable, maybe it isn’t. Now what? Computer scientists have come up with two basic approaches to solv- ing this question, relying on different data structures and different meth- ods of making the decision. One is called access control lists and the other is called capabilities. It’s actually a little inaccurate to claim that computer scientists came up with these approaches, since they’ve been in use in non-computer contexts for millennia. Let’s look at them in a more general perspective before we consider operating system implementa- tions. Let’s say we want to start an exclusive nightclub (called, perhaps, Chez Andrea3) restricted to only the best operating system researchers and developers. We don’t want to let any of those database or program- ming language people slip in, so we’ll need to make sure only our ap- proved customers get through the door. How might we do that? One 2Almost. Remember the bits in the page table that determine whether a particular page can be read, written, or executed? But it’s not the operating system doing the runtime check here, it’s the virtual memory hardware. 3The authors Arpaci-Dusseau would like to note that author Reiher is in charge of these name choices for the security chapters, and did not strong-arm him into using their names throughout this and other examples. We now return you to your regular reading... © 2008–23, ARPACI-DUSSEAU (OSTEP)',
        '12 DISTRIBUTED SYSTEM SECURITY will continue to assume that the authentication is valid. It won’t ask for your password every time you click a link or perform some other inter- action with it. (And a good thing, too. Imagine how much of a pain it would be if you had to provide your password every time you wanted to do anything.) If your session is encrypted at this point, it could regard your proper use of the cryptography as a form of authentication; but you might even be able to quit your web browser, start it up again, navigate back to that web site, and still be treated as an authenticated user, with- out a new request for your password. At that point, you’re no longer using the same cryptography you used before, since you would have es- tablished a new session and set up a new cryptographic key. How did your partner authenticate that you were the one receiving the new key? In such cases, the site you are working with has chosen to make a se- curity tradeoff. It veriﬁed your identity at some time in the past using your password and then relies on another method to authenticate you in the future. A common method is to use web cookies. Web cookies are pieces of data that a web site sends to a client with the intention that the client stores that data and send it back again whenever the client next communicates with the server. Web cookies are built into most browsers and are handled invisibly, without any user intervention. With proper use of cryptography, a server that has veriﬁed the password of a client can create a web cookie that securely stores the client’s identity. When the client communicates with the server again, the web',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
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

* Size: 540 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                           | label                                                          |
  |:---------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                             | string                                                                               | float                                                          |
  | modality | text                                                                               | text                                                                                 |                                                                |
  | details  | <ul><li>min: 10 tokens</li><li>mean: 20.63 tokens</li><li>max: 37 tokens</li></ul> | <ul><li>min: 33 tokens</li><li>mean: 292.88 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.54</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                   | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:-----------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>an approach which tries to segregate hot and cold seg- ments</code>                                                    | <code>an approach which tries to segregate hot and cold seg- ments. A hot segment is one in which the contents are being frequently over-written; thus, for such a segment, the best policy is to wait a long time before cleaning it, as more and more blocks are getting over-written (in new segments) and thus being freed for use. A cold segment, in con- trast, may have a few dead blocks but the rest of its contents are relatively stable. Thus, the authors conclude that one should clean cold segments sooner and hot segments later, and develop a heuristic that does exactly that. However, as with most policies, this policy isn’t perfect; later ap- proaches show how to do better [MR+97]. © 2008–23, ARPACI-DUSSEAU THREE EASY PIECES</code>                                                                                                                                                                                                                                                                                      | <code>1.0</code> |
  | <code>8 LOG-STRUCTURED FILE SYSTEMS 43.8 What About Directories</code>                                                       | <code>8 LOG-STRUCTURED FILE SYSTEMS 43.8 What About Directories? Thus far, we’ve simpliﬁed our discussion a bit by only considering in- odes and data blocks. However, to access a ﬁle in a ﬁle system (such as /home/remzi/foo, one of our favorite fake ﬁle names), some directo- ries must be accessed too. So how does LFS store directory data? Fortunately, directory structure is basically identical to classic UNIX ﬁle systems, in that a directory is just a collection of (name, inode number) mappings. For example, when creating a ﬁle on disk, LFS must both write a new inode, some data, as well as the directory data and its inode that refer to this ﬁle. Remember that LFS will do so sequentially on the disk (after buffering the updates for some time). Thus, creating a ﬁle foo in a directory would lead to the following new structures on disk: I[k] b[0]:A0 (foo, k) I[dir] b[0]:A2 m[k]:A1 m[dir]:A3 Dk Ddir imap A0 A1 A2 A3 The piece of the inode map contains the information for the location of both the...</code> | <code>1.0</code> |
  | <code>Kernel — Scheduling  The dispatcher uses a 32-level priority scheme to determine the order of thread execution</code> | <code>Windows Scheduling  Windows uses priority-based preemptive scheduling  Highest-priority thread runs next  Dispatcher is scheduler  Thread runs until (1) blocks, (2) uses time slice, (3) preempted by higher-priority thread  Real-time threads can preempt non-real-time  32-level priority scheme  Variable class is 1-15, real-time class is 16-31  Priority 0 is memory-management thread  Queue for each priority  If no run-able thread, runs idle thread 5.65 Silberschatz, Galvin and Gagne ©2018 Operating System Concepts – 10th Edition</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 1

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
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Time
- **Training**: 18.2 seconds

### Framework Versions
- Python: 3.12.14
- Sentence Transformers: 6.0.1
- Transformers: 5.17.0
- PyTorch: 2.14.0
- Accelerate: 1.15.0
- Datasets: 5.0.1
- Tokenizers: 0.23.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

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