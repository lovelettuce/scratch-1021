# Generative Recommendation with Semantic IDs (GRID)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/config-hydra-89b8cd)](https://hydra.cc/)
[![Lightning](https://img.shields.io/badge/pytorch-lightning-792ee5)](https://lightning.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.22224-b31b1b.svg)](https://arxiv.org/abs/2507.22224)


**GRID** (Generative Recommendation with Semantic IDs) is a state-of-the-art framework for generative recommendation systems using semantic IDs, developed by a group of scientists and engineers from [Snap Research](https://research.snap.com/team/user-modeling-and-personalization.html). This project implements novel approaches for learning semantic IDs from text embedding and generating recommendations through transformer-based generative models.

## 🚀 Overview

GRID facilitates generative recommendation three overarching steps:

- **Embedding Generation with LLMs**: Converting item text into embeddings using any LLMs available on Huggingface. 
- **Semantic ID Learning**: Converting item embedding into hierarchical semantic IDs using Residual Quantization techniques such as RQ-KMeans, RQ-VAE, RVQ. 
- **Generative Recommendations**: Using transformer architectures to generate recommendation sequences as semantic ID tokens. 


## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/snap-research/GRID.git
cd GRID

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Data Preparation

Prepare your dataset in the expected format:
```
data/
├── train/       # training sequence of user history 
├── validation/  # validation sequence of user history 
├── test/        # testing sequence of user history 
└── items/       # text of all items in the dataset
```

We provide pre-processed Amazon data explored in the [P5 paper](https://arxiv.org/abs/2203.13366) [4]. The data can be downloaded from this [google drive link](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing).

### 2. Embedding Generation with LLMs

Generate embeddings from LLMs, which later will be transformed into semantic IDs. 

```bash
python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty # avaiable data includes 'beauty', 'sports', and 'toys'
```

### 3. Train and Generate Semantic IDs

Learn semantic ID centroids for embeddings generated in step 2:

```bash
python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<output_path_from_step_2>/merged_predictions_tensor.pt \ # this can be found in the log dirs in step2
    embedding_dim=2048 \ # the model dimension of the LLMs you use in step 2. 2048 for flan-t5-xl as used in this example.
    num_hierarchies=3 \  # we train 3 codebooks
    codebook_width=256 \ # each codebook has 256 rows of centroids  
```


SID tokenizer training defaults follow [GRID, Section 4](https://arxiv.org/html/2507.22224v1#S4):

| Experiment | Training budget | Optimizer | Learning rate |
| --- | --- | --- | --- |
| `rkmeans_train_flat` | 1,000 steps per layer (3,000 with 3 layers) | SGD | 0.5 |
| `rvq_train_flat` | 1,000 steps per layer (3,000 with 3 layers) | Adam | 0.001 |
| `rqvae_train_flat` | 15,000 total steps | Adagrad | 0.001 |

All three default to 3 codebooks of 256 entries, 8 GPUs, and batch size 2,048
per GPU (global batch size 16,384). Input embedding paths and dimensions remain
required. Training stops at the step budget, with no epoch cap or early stopping;
the checkpoint is saved at the final step budget. RK-Means and RVQ normalize
residuals; RQ-VAE uses the existing BatchNorm + L2 input normalization and jointly
trains the autoencoder and codebooks. The paper describes RQ-VAE preprocessing as
whitening; this preset retains the repository's implementation. SGD with learning
rate 0.5 for RK-Means comes from the implementation; the paper specifies the
optimizers and learning rates for RVQ and RQ-VAE.

Override `trainer.devices=1` to run on one GPU, or set it to another available
GPU count. With batch size unchanged, this changes the global batch size relative
to the paper. Changing `num_hierarchies` automatically adjusts the RK-Means/RVQ
step budget to preserve 1,000 steps per layer. Historical configurations and
checkpoints under `outputs/` retain their original training budgets.

Generate SIDs:

```bash
python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<output_path_from_step_2>/merged_predictions_tensor.pt \ 
    embedding_dim=2048 \ 
    num_hierarchies=3 \  
    codebook_width=256 \ 
    ckpt_path=<the_checkpoint_you_just_get_above> # this can be found in the log dir for training SIDs
```


### 4. Train Generative Recommendation Model with Semantic IDs

Train the recommendation model using the learned semantic IDs:

```bash
python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/beauty \ 
    semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 # Please note that we add 1 for num_hierarchies because in the previous step we appended one additional digit to de-duplicate the semantic IDs we generate.
```

### 4. Generate Recommendations

Run inference to generate recommendations:

```bash
python -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/beauty \ 
    semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
    ckpt_path=<the_checkpoint_you_just_get_above> \ # this can be found in the log dir for training GR models
    num_hierarchies=4 \ # Please note that we add 1 for num_hierarchies because in the previous step we appended one additional digit to de-duplicate the semantic IDs we generate.
```

### Beauty on A100: finite-epoch BF16 training (single or multiple GPUs)

Use `tiger_train_beauty_a100` after generating the four-digit SID tensor. This
preset inherits `tiger_train_flat`, keeps its model, optimizer, learning rate,
per-device batch settings and full validation/test datasets, and defaults to
20 finite epochs with one full validation at each epoch end. The training
progress bar now has a bounded batch total for the current rank; `global_step`
continues to count optimizer updates and is retained in logs and checkpoints.

The raw batch of 32 user histories is augmented into up to 512 subsequences, so
its actual size varies. Training data is read once per epoch, then persistent
workers reset and shuffle their assigned files for the next epoch. Preprocessing
can filter individual records, so the displayed total is an upper-bound estimate
from the raw record count rather than a guarantee of emitted batches.

The preset uses BF16 mixed precision (`trainer.precision=bf16-mixed`) on A100.
Override with `trainer.precision=32-true` for FP32. Keep the product of GPU count
and accumulation steps at 16 to approximately preserve the global effective
batch: 1 GPU/16 accumulations, 2 GPUs/8 accumulations, or 4 GPUs/4 accumulations.
Changing this product changes the optimization budget and does not automatically
require the same learning rate.

Run from the project root, replacing the data/SID paths if necessary:

```bash
DATA_DIR="$PWD/data/amazon_data/beauty"
SID_PATH="$PWD/outputs/beauty_rkmeans_inference/pickle/merged_predictions_tensor.pt"

# Single GPU.
CUDA_VISIBLE_DEVICES=0 python -m src.train \
  experiment=tiger_train_beauty_a100 \
  data_dir="$DATA_DIR" semantic_id_path="$SID_PATH" num_hierarchies=4 \
  gpu_count=1 accumulation_steps=16 \
  hydra.run.dir=outputs/beauty_tiger_a100_1gpu

# Two GPUs. Lightning launches local DDP processes; do not add torchrun.
CUDA_VISIBLE_DEVICES=0,1 python -m src.train \
  experiment=tiger_train_beauty_a100 \
  data_dir="$DATA_DIR" semantic_id_path="$SID_PATH" num_hierarchies=4 \
  gpu_count=2 accumulation_steps=8 \
  hydra.run.dir=outputs/beauty_tiger_a100_2gpu

# Four GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train \
  experiment=tiger_train_beauty_a100 \
  data_dir="$DATA_DIR" semantic_id_path="$SID_PATH" num_hierarchies=4 \
  gpu_count=4 accumulation_steps=4 \
  hydra.run.dir=outputs/beauty_tiger_a100_4gpu
```

Override `trainer.max_epochs` to change the finite training limit. The inherited
early-stopping patience is ten validation checks, so a 20-epoch run can stop
earlier after ten non-improving epoch-end validations. Training partitions source files across DataLoader workers and, in multi-GPU
runs, across ranks. Beauty has 175 files per split, enough for the documented
1/2/4-GPU combinations with eight workers per rank. Other datasets with too
few files may fall back to file replication. Variable augmented batch sizes and
different file assignments mean multi-GPU runs are not numerically identical to
single-GPU runs.
Retrieval metrics aggregate metric sums and sample counts across ranks before
checkpoint/early-stopping monitoring.

To resume, append `ckpt_path=/absolute/path/to/model.ckpt` and use a new output
directory. Lightning restores model, optimizer, global step, epoch and callback
state. `trainer.max_epochs=20` is the total epoch target, not 20 additional
epochs; prefer the same GPU count and accumulation when resuming. The iterable
loader does not restore an exact intra-epoch cursor. Verify the restored epoch,
global step and next epoch-end validation in logs. Check `nvidia-smi` before
selecting GPUs: BF16 does not remove contention or guarantee freedom from OOM.

## Supported Models:

### Semantic ID:

1. Residual K-means proposed in One-Rec [2]
2. Residual Vector Quantization
3. Residual Quantization with Variational Autoencoder [3]

### Generative Recommendation:

1. TIGER [1]

## 📚 Citation

If you use GRID in your research, please cite:

```bibtex
@inproceedings{grid,
  title     = {Generative Recommendation with Semantic IDs: A Practitioner's Handbook},
  author    = {Ju, Clark Mingxuan and Collins, Liam and Neves, Leonardo and Kumar, Bhuvesh and Wang, Louis Yufeng and Zhao, Tong and Shah, Neil},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2025}
}
```

## 🤝 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Configuration management by [Hydra](https://hydra.cc/)
- Inspired by recent advances in generative AI and recommendation systems
- Part of this repo is built on top of https://github.com/ashleve/lightning-hydra-template

## 📞 Contact

For questions and support:
- Create an issue on GitHub
- Contact the development team: Clark Mingxuan Ju (mju@snap.com), Liam Collins (lcollins2@snap.com), Bhuvesh Kumar (bhuvesh@snap.com) and Leonardo Neves (lneves@snap.com).

## Bibliography 

[1] Rajput, Shashank, et al. "Recommender systems with generative retrieval." Advances in Neural Information Processing Systems 36 (2023): 10299-10315.

[2] Deng, Jiaxin, et al. "Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment." arXiv preprint arXiv:2502.18965 (2025).

[3] Lee, Doyup, et al. "Autoregressive image generation using residual quantization." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[4] Geng, Shijie, et al. "Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5)." Proceedings of the 16th ACM conference on recommender systems. 2022.

## Long-tail recommendation evaluation

See [the long-tail evaluation guide](docs/long_tail_evaluation.md) for frozen
training-popularity groups, resumable Beauty evaluation with beam 10/20/50,
teacher-forced SID diagnostics, and offline report regeneration in the `grid`
conda environment. Default configuration: `configs/analysis/beauty_long_tail.yaml`.

Second experiment: [second-layer search attribution guide](docs/second_layer_analysis.md)
adds training support, potential target-window counts, frequency baselines and
native-score pruning attribution while reusing the first experiment's frozen artifacts.
Configuration: `configs/analysis/beauty_second_layer.yaml`.

Beauty/Toys × RQ-kmeans/RQ-VAE 的 seed42 交叉验证使用
[`scripts/run_rq_cross_validation.sh`](scripts/run_rq_cross_validation.sh)，包含 Beauty + RQ-kmeans 的 SID 与 GR 重训。
SID 与 GR 均使用GPU0/1/2，最多三组并行、第四组排队，每组独占一张卡；SID保留默认训练步数和每卡batch。配置、恢复和离线重算见 [`docs/rq_cross_validation.md`](docs/rq_cross_validation.md)。
