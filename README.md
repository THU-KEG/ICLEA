# ICLEA: Interactive Contrastive Learning for Self-Supervised Entity Alignment

Code for the CIKM 2022 paper [Interactive Contrastive Learning for Self-Supervised Entity Alignment](https://arxiv.org/abs/2201.06225).

[[Paper](https://arxiv.org/abs/2201.06225)] [[DOI](https://doi.org/10.1145/3511808.3557364)] [[Reproducibility notes](REPRODUCIBILITY.md)]

ICLEA aligns entities across two knowledge graphs without seed alignments. It combines pretrained name and description features, relation-aware neighborhood aggregation, momentum contrastive learning, and pseudo-aligned entity pairs mined during training.

This repository contains a cleaned version of the research implementation. It keeps the behavior of the released model under `author-code` and provides a separate `paper-exact-rgat` implementation for the RGAT equations described in the paper.

## Installation

The verified environment uses Python 3.7.9, PyTorch 1.8.0, CUDA 11.1, and Faiss 1.7.0.

```bash
conda env create -f environment.yml
conda activate iclea-py37
```

The training dependencies are also listed in `requirements.txt`. Text preprocessing has a separate dependency file:

```bash
pip install -r requirements-preprocess.txt
```

## Data preparation

Place the prepared DBP15K files under `data/DBP15K`:

```text
data/
└── DBP15K/
    ├── zh_en/
    ├── ja_en/
    └── fr_en/
```

The repository does not include datasets or generated feature files. The loader expects the graph files and LaBSE/description pickle files in each language directory.

To build the text features from raw files, prepare three UTF-8 TSV files for each KG side:

```text
entity_names_1.tsv    HY_entity_id<TAB>raw URI or entity name
relation_names_1.tsv  JAPE_relation_id<TAB>raw URI or relation name
descriptions_1.tsv    HY_entity_id<TAB>description text
```

Then run:

```bash
python scripts/preprocess_text.py \
  --entity-names raw/zh_en/entity_names_1.tsv \
  --relation-names raw/zh_en/relation_names_1.tsv \
  --descriptions raw/zh_en/descriptions_1.tsv \
  --output-dir data/DBP15K/zh_en \
  --side 1 \
  --setting original \
  --device cuda:0
```

Run the command for both KG sides. The script writes the pickle files consumed by the loader and a manifest containing the model revision, counts, and file hashes. The paper does not specify the exact SentenceTransformers revision used for the archived description features, so newly generated files should be identified by their manifest.

## Quick start

Run one DBP15K experiment with:

```bash
GPU_IDS=0 bash scripts/reproduce.sh zh_en original paper
```

The positional arguments are:

```text
scripts/reproduce.sh {zh_en|ja_en|fr_en} [original|translated] [paper|top1-no-threshold]
```

The launcher trains for 300 epochs with seed 37. Results are written to `out/results`, checkpoints to `out/checkpoints`, and logs to `out/logs`. These directories are ignored by Git.

For five seeds on one language pair:

```bash
python scripts/run_five.py \
  --language zh_en \
  --setting original \
  --profile paper \
  --gpus 0,1,2,3
```

For all three DBP15K language pairs:

```bash
python scripts/run_three.py \
  --setting original \
  --profile paper \
  --gpus 0,1,2,3
```

The launchers in this snapshot accept physical GPU IDs 0 to 3 and expose one GPU to each training process.

These commands use the repository defaults. The tuned configuration associated with the recovery table below has dedicated launchers.

## Training options

Pseudo-pair mining is selected with `--profile`:

| Profile | Retrieval | Squared L2 threshold | ICL warm-up |
|---|---|---:|---:|
| `paper` | bidirectional Faiss Top-1 over both KGs | `< 1.0` | 0 |
| `top1-no-threshold` | bidirectional Faiss Top-1 over both KGs | none | 0 |

RGAT behavior is selected with `--rgat_impl`:

| Implementation | Neighbor input | Description |
|---|---:|---|
| `author-code` | center plus up to 14 neighbors | preserves the released computation |
| `paper-exact-rgat` | center plus up to 15 neighbors | follows the attention and activation equations in the paper |

`author-code` is the default. The selected profile and RGAT implementation are stored in every result JSON.

## Evaluation

The evaluator ranks test entities against the complete target KG using Faiss squared L2. A saved checkpoint can be evaluated independently with:

```bash
python scripts/evaluate_checkpoints.py \
  --checkpoint out/checkpoints/<run_name>/best_joint_hits10.pt \
  --candidate-scope full-target \
  --weight-source online \
  --output out/diagnostics/<run_name>_eval.json
```

`verify_goal88.py` checks the ZH_EN Hits@1 target. `verify_joint_goal.py` checks a joint Hits@1 and Hits@10 target from one independently evaluated checkpoint.

## Reproduced results

The table below records the paper-result recovery snapshot in the original multilingual setting. JA_EN and FR_EN report the mean and sample standard deviation over seeds 37 to 41. ZH_EN is one independently re-evaluated seed-37 checkpoint.

| Dataset | Paper Hits@1 | Paper Hits@10 | Recovered Hits@1 | Recovered Hits@10 |
|---|---:|---:|---:|---:|
| ZH_EN | 88.4 | 97.2 | 88.895 | 97.219 |
| JA_EN | 91.9 | 97.5 | 92.347 +/- 0.271 | 97.836 +/- 0.135 |
| FR_EN | 98.6 | 99.9 | 99.038 +/- 0.051 | 99.918 +/- 0.013 |

Run the same recovery configurations with:

```bash
export ICLEA_PYTHON="$CONDA_PREFIX/bin/python"

GPU_IDS=0,1,2,3 CONTROL_TAG=paper_recovery \
  bash scripts/launch_zh_hits10_description_scale.sh

python scripts/launch_cross_dataset_current.py \
  --tag paper_recovery_cross_dataset \
  --gpus 0,1,2,3
```

These runs use a tuned test-best protocol: checkpoints are selected with test metrics, and the recovered configuration changes the description weight and learning rate at epoch 151. The table is therefore a record of result recovery, not a test-blind estimate. The exact configuration, per-run evidence, report hashes, and differences from the paper are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Implementation notes

The loader keeps neighbor entities and their relations in the same slots and masks padded slots in relation-gated attention. The momentum encoder starts from the online encoder parameters. Evaluation uses the full target KG and restores the previous train/eval state. The training loop also avoids retaining graphs that are no longer needed.

The repository keeps experimental switches for reconstructing released-code behavior and for the parameter studies used during reproduction. They are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md); the main README only covers the standard entry points.

## Repository layout

```text
loader/                  DBP15K data loader
model/                   ICLEA model and training loop
script/preprocess/       original preprocessing utilities
scripts/                 launch, evaluation, and verification tools
tests/                   unit tests
run.py                   training entry point
REPRODUCIBILITY.md       protocol and evidence record
```

## Citation

If you use ICLEA, please cite:

```bibtex
@inproceedings{zeng2022interactive,
  author    = {Kaisheng Zeng and Zhenhao Dong and Lei Hou and Yixin Cao and
               Minghao Hu and Jifan Yu and Xin Lv and Lei Cao and Xin Wang and
               Haozhuang Liu and Yi Huang and Junlan Feng and Jing Wan and
               Juanzi Li and Ling Feng},
  title     = {Interactive Contrastive Learning for Self-Supervised Entity Alignment},
  booktitle = {Proceedings of the 31st ACM International Conference on
               Information and Knowledge Management},
  pages     = {2465--2475},
  year      = {2022},
  doi       = {10.1145/3511808.3557364}
}
```

## Acknowledgements

Special thanks to [Haoyun Hong](https://github.com/HaoyunHong) for her generous help during the early stages of the ICLEA project.
