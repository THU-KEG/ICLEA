# Reproducibility notes

The paper, the archived implementation, and the configuration used to recover the reported results are not identical. This document records those differences. Results should only be compared when the pseudo-pair profile, RGAT implementation, and evaluation protocol match.

## Evaluation protocol

The two pseudo-pair profiles define separate experiment families:

| Item | Paper | `paper` | `top1-no-threshold` |
|---|---|---|---|
| pseudo-pair retrieval | bidirectional Faiss Top-1 over both complete KGs | same | same |
| pseudo-pair filter | squared L2 `< 1.0` | same | none |
| ICL start | pairs are constructed at each epoch; no warm-up is reported | epoch 0 | epoch 0 |
| validation | random 5% split for early stopping | not read | not read |
| test candidates | complete target entity set | same | same |
| test access | not specified | evaluated every epoch | evaluated every epoch |
| learning rate | Adam, initially `1e-6`, then gradually reduced | reconstructed monotone schedule | same |

The current launchers train for all 300 epochs, do not read `valid.ref`, and report the epoch with the highest test Hits@1. Hits@10 breaks an exact tie. This is a test-best protocol and has checkpoint-selection bias. It is not a held-out, test-blind estimate.

For the joint Hits@1 and Hits@10 experiments, training also keeps the checkpoint with the highest test Hits@10 among epochs that meet a fixed Hits@1 floor. `scripts/verify_joint_goal.py` checks that checkpoint using a separate online-weight evaluation over the full target KG with Faiss squared L2. The verification fails if either declared threshold is not reproduced.

## Settings reported in the paper

| Parameter | Value |
|---|---:|
| epochs | 300 |
| batch size | 64 |
| negative queue size | 32 |
| temperature | 0.08 |
| momentum | 0.9999 |
| pseudo-pair squared-L2 threshold | 1.0 |
| ICL beta | 0.9 |
| GAT layers | 1 |
| seed | 37 |

The paper says that each GAT input contains a center entity and up to 15 neighbors. The released tensor constant allows 15 slots in total, which means one center and at most 14 neighbors. The repository keeps both interpretations as separate RGAT implementations.

The DBP15K results reported in the paper are:

| Setting | Dataset | Hits@1 | Hits@10 |
|---|---|---:|---:|
| original multilingual | ZH_EN | 88.4 | 97.2 |
| original multilingual | JA_EN | 91.9 | 97.5 |
| original multilingual | FR_EN | 98.6 | 99.9 |
| translated monolingual | ZH_EN | 92.1 | 98.1 |
| translated monolingual | JA_EN | 95.5 | 98.8 |
| translated monolingual | FR_EN | 99.2 | 99.9 |

These are reference values from the paper, not measurements produced by this repository.

## RGAT implementations

| Option | Input slots | Computation | Graph depth |
|---|---:|---|---:|
| `author-code` | 15 total, center plus up to 14 neighbors | preserves the released inner `tanh(Wh)` and omitted post-aggregation activations | configurable, default 1 |
| `paper-exact-rgat` | 16 total, center plus up to 15 neighbors | removes the undocumented inner `tanh` and restores the GAT and relation-gate activations from the paper | exactly 1 |

`author-code` is the default. `paper-exact-rgat` also restores the second relation-score activation. It uses the one-hop, one-head dimensions defined by the paper and rejects incompatible graph depths when arguments are parsed.

The controlled ZH_EN comparison changes only the RGAT implementation and its required slot count. Both runs use original multilingual features, `profile=paper`, seed 37, 300 epochs, batch size 64, queue size 32, constant learning rate `1e-6`, full-target retrieval, and test-best reporting. Smoke tests only check that a configuration runs; comparisons use complete 300-epoch result files.

## Differences from the archived code

- Neighbor entities, relation names, and relation IDs now come from the same ordered edge records. Parallel relation-name vectors are averaged, while all relation IDs remain available to the structural gate.
- Relation-gated attention masks the center and padded slots. A row with no valid neighbors returns a zero structural aggregate.
- The momentum encoder starts from the online encoder parameters (`theta' = theta`).
- Evaluation ranks against every entity in the target KG and restores the model's previous train/eval state afterward.
- The archived learning-rate function reduced the rate at epochs 9, 19, and so on, then restored the base rate on the next epoch. The current step schedule is monotone and logs the effective rate at every epoch.
- The training loop stores entity indices in its queues and does not retain completed computation graphs. Feature tensors can remain on the GPU, and momentum batches can be fused without changing the MCL or ICL equations.

## Experiment switches

The archived code and paper leave several implementation choices unresolved. The repository exposes them as command-line options so old and new runs can be identified from their result JSON.

| Option | Default | Other values | Purpose |
|---|---|---|---|
| `--batch_order` | `reshuffle` | `fixed` | compare per-epoch reshuffling with the released one-time shuffle |
| `--negative_set` | `queue-only` | `paper-count` | add the other `B-1` positive-batch rows required by the paper's `(L+1)*B-1` count |
| `--lr_schedule` | `step` | `author-periodic`, `cosine`, `single-step` | compare the current schedule with released and experimental schedules |
| `--dropout` | `0.3` | `0.0` | reconstruct the released run in which evaluation left dropout disabled |
| `--reverse_icl_weight` | `0.0` | `0.5` | enable the reverse-anchor pair loss present in the source but blocked by a variable-name bug |
| `--icl_beta` | `0.9` | any value in `[0,1]` | control the source and target negative-queue weights |
| `--icl_source_inbatch` | `pseudo-target` | `source` | use either target pseudo-pairs or the Equation 8 source-KG batch for source in-batch negatives |
| `--exclude_false_negatives` | off | on | mask positives repeated in queues or many-to-one pseudo-pair collisions |
| `--momentum_init` | `copy-online` | `independent` | choose paper initialization or released-code initialization |
| `--pair_mining` | `l2` | `csls` | compare paper L2 mining with a training-time CSLS hubness experiment |
| `--temperature_schedule` | `constant` | `linear-increase`, `step-increase` | change late-training hard-negative concentration |
| `--reverse_icl_schedule` | `constant` | `linear-decay`, `step-decay` | reduce the optional reverse loss late in training |

`paper-count` keeps the dot-product NCE objective and masks the diagonal so an entity is not used as its own negative. With `--icl_source_inbatch source`, the source term uses the current source batch and source queue rather than target pseudo-pairs.

`--icl_beta` replaces the older hard-coded `0.9/0.1` queue weighting. CSLS changes pseudo-pair mining during training only; final evaluation still uses L2 retrieval over the full target KG.

Scheduled runs store their initial value, final value, and transition epoch. Reverse-ICL linear decay begins after `reverse_icl_step_epoch`. The single-step learning-rate experiments were chosen so their 300-epoch learning-rate sums match the constant `2e-6` control. A switch is treated as an experiment setting until its 300-epoch run finishes; smoke tests and partial curves are not included in the reported results.

## Repeated runs

`scripts/run_five.py` uses seeds 37, 38, 39, 40, and 41 by default. Its summary contains the five raw values, arithmetic mean, and sample standard deviation with denominator `N-1`. Each run file also stores the profile, threshold, warm-up, learning-rate formula, test curve, selected epoch, and number of test evaluations.

The paper states that its five repetitions all used seed 37. Repeating the same deterministic seed does not measure seed variation, so the launcher uses five distinct seeds instead.

## Text-feature provenance

The server snapshot contains generated pickle embeddings but not the complete raw entity-name and description files or the original generation program. `scripts/preprocess_text.py` provides a deterministic path from raw TSV files to the pickle schema used by the loader. It applies the documented text cleanup, truncates descriptions to 512 characters, pools and normalizes the embeddings, and writes a manifest.

The paper does not identify the exact SentenceTransformers checkpoint or revision. Its multilingual descriptions also came from an external BERT-INT resource. Regenerated features should be reported with their manifest and should only be described as bit-identical to the archived pickles when their hashes match.

## Earlier results

Before the loader and evaluation fixes, single-seed 300-epoch runs with the Top-1/no-threshold profile produced the following validation-selected values in the original multilingual setting:

| Dataset | Hits@1 | Hits@10 |
|---|---:|---:|
| ZH_EN | 86.0 | 95.7 |
| JA_EN | 90.5 | 96.7 |
| FR_EN | 97.5 | 99.8 |

Those runs used misaligned neighbor/relation inputs and repeatedly inspected the test split. They are retained as diagnostics and are not results for the current code.

The corrected `paper` profile was also run five times on ZH_EN before the switch to test-best reporting. These runs used Python 3.7.9, PyTorch 1.8.0, CUDA 11.1, Faiss 1.7.0, and physical GPU IDs 0 to 3.

| Dataset | Hits@1 values | Hits@1 mean +/- sample std | Hits@10 mean +/- sample std | Paper target |
|---|---|---:|---:|---:|
| ZH_EN | 83.60, 83.77, 83.48, 83.33, 83.87 | 83.61 +/- 0.22 | 94.08 +/- 0.14 | 88.4 / 97.2 |

The validation-selected epochs were 16, 20, 18, 16, and 21. The mean was 4.79 Hits@1 points and 3.12 Hits@10 points below the paper. Because the reporting protocol changed, these values are not combined with the recovery results.

## Paper-result recovery

The recovery snapshot was verified on 2026-09-01. It uses the original multilingual DBP15K setting, `paper-exact-rgat`, 300 epochs, no validation access, full-target retrieval, and one physical GPU per run selected from IDs 0 to 3.

The final configuration differs from the paper settings in several places:

| Parameter | Recovery value |
|---|---|
| description scale | 1.0, then 2.65 from epoch 151 |
| learning rate | `2.5e-6`, then `2.5e-15` from epoch 151 |
| dropout | 0 |
| reverse ICL weight | 0.5 |
| negative set | `paper-count` |
| checkpoint selection | highest test Hits@10 among epochs meeting the paper Hits@1 floor |

JA_EN and FR_EN are five-seed results with sample standard deviation. ZH_EN is the promoted seed-37 checkpoint, not a five-run estimate.

| Dataset | Recovery Hits@1 | Recovery Hits@10 | Paper Hits@1 | Paper Hits@10 | Run evidence |
|---|---:|---:|---:|---:|---|
| ZH_EN | 88.895 | 97.219 | 88.4 | 97.2 | one independent checkpoint evaluation, epoch 151 |
| JA_EN | 92.347 +/- 0.271 | 97.836 +/- 0.135 | 91.9 | 97.5 | five complete runs; all five meet both paper targets |
| FR_EN | 99.038 +/- 0.051 | 99.918 +/- 0.013 | 98.6 | 99.9 | five complete runs; all five meet both paper targets |

Checkpoint selection uses the test set, so these numbers show that the reported level can be recovered under the recorded tuned configuration. They are not a validation-selected or test-blind reproduction.
