# Domain Transfer becomes Identifiable via a Single Alignment

This repository contains the official code accompanying the paper *"Domain Transfer becomes Identifiable via a Single Alignment"* published in ICML 2026. We provide two self-contained experiment suites:

1. **`synthetic/`** — Controlled identifiability experiments on synthetic data .
2. **`image/`** — Training and evaluation code for image to image translation.

---

## Repository Structure

```
.
├── synthetic/                         # Synthetic identifiability experiments
│   ├── synth_exp.py                   # Main training script
│   ├── autoencoder.py                 # MLP encoder/decoder + regularizers
│   ├── data.py                        # Synthetic data generators (PNL / sparse_pnl / MLP)
│   ├── eval.py                        # MCC, R^2, scatter plots
│   ├── models.py                      # Discriminator network (for GAN matching)
│   ├── utils.py                       # Seeding + Hungarian matching
│   └── scripts/
│       └── run_nica_gan_anchor.sh     # Reference launcher for the main experiment
│
├── image/            
│   ├── train.py                       # Main training / evaluation entry point
│   ├── model.py                       # Generator and discriminator architectures
│   ├── configs/
│   │   ├── mnist.yaml                 # Default config (GAN + anchor + Jacobian reg.)
│   │   └── mnist_wo_jac_reg.yaml      # Same setup but with Jacobian reg. disabled
│   ├── src/
│   │   ├── trainer.py                 # Core training loop / losses
│   │   └── trainer_utils.py           # Model factory helpers
│   └── utils/
│       ├── data_loader.py             # Dataset / paired-loader utilities
│       ├── data.py                    # Image folder helpers
│       ├── losses.py                  # GAN / Jacobian / R1 losses
│       └── prepare_mnist_dataset.py   # MNIST → trainA/trainB/testA/testB preparation
│
├── requirements.txt
└── README.md
```

---

## Installation

We recommend Python ≥ 3.10 with a CUDA-enabled PyTorch build.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

[Weights & Biases](https://wandb.ai/) is used for logging in both demos. Either run `wandb login` before training, or set `use_wandb: False` (image demo) / pass `WANDB_MODE=offline` (synthetic) to disable online logging.

---

## 1. Synthetic Experiments

The synthetic suite verifies the identifiability claims of the paper on data with known ground-truth translation. 

### 1.1 Reproducing the main experiment

The reference experiment trains the model with GAN matching plus a *single* anchor pair on the dependent-uniform sparse PNL setup. From inside `synthetic/`:

```bash
cd synthetic
bash scripts/run_nica_gan_anchor.sh
```


---

## 2. Image Translation 

The demo config trains a one-sided translator from MNIST to 90°-rotated MNIST using a GAN loss, a single paired anchor, and (optionally) Jacobian regularization implemented as a sparse finite-difference proxy.

### 2.1 Prepare the dataset

```bash
cd image_translation_demo
python utils/prepare_mnist_dataset.py --output_dir ./data/rotatedmnist
```

This produces:

```
data/rotatedmnist/
├── trainA/          # Original MNIST
├── trainB/          # Rotated MNIST
├── testA/           # Original MNIST test
├── testB/           # Rotated MNIST test
├── trainA_attr.csv  # Labels for trainA
├── trainB_attr.csv  # Labels for trainB
├── testA_attr.csv   # Labels for testA
└── testB_attr.csv   # Labels for testB
```

The labels csv files are used by the baseline method DIMENSION based on auxiliary variables. For the proposed method, the labels can be a dummy variable with all ones.

### 2.2 Configure training

Edit `configs/mnist.yaml` and update `data_path` to point at the dataset you just prepared:

```yaml
data_path: ./data/rotatedmnist
```

Highlights of the default config:

- `network_type: fcn` — fully-connected network (good for 32×32 inputs).
- `one_sided: True` — only train the A → B translator.
- `paired_loss_w: 1.0` and `paired_max_pairs: 1` — single-anchor supervision.
- `jacobian_loss_w: 0.01` with `jacobian_norm_type: fd_l1` — sparse finite-difference Jacobian regularization. Set `jacobian_loss_w: 0` to disable.
- `use_wandb: True` — set to `False` to disable W&B logging.

### 2.3 Train

```bash
python train.py --config configs/mnist.yaml
```

Common flags:

- `--debug` — disables W&B logging and uses minimal data, useful for smoke-testing.
- `--resume` — resume from `results/models/<run_name>/checkpoint-current.pt`.

Outputs:

- `results/models/<run_name>/` — model checkpoints.
- `results/samples/<run_name>/` — periodic translation samples and debug paired samples.

### 2.4 Evaluate

```bash
python train.py \
    --config configs/mnist.yaml \
    --eval \
    --checkpoint_path results/models/mnist_demo/checkpoint-current.pt
```

This loads the checkpoint and writes a grid of test-set translations to the sample directory.

### 2.5 With vs. without Jacobian regularization

Two configs are provided so the effect of Jacobian regularization can be reproduced directly:

```bash
# With Jacobian regularization (jacobian_loss_w = 0.01)
python train.py --config configs/mnist.yaml

# Without Jacobian regularization (jacobian_loss_w = 0)
python train.py --config configs/mnist_wo_jac_reg.yaml
```

### 2.6 Configuration reference

**Network architecture**

| Option | Values | Description |
|---|---|---|
| `network_type` | `fcn`, `cnn` | FCN for small images, CNN for larger images |
| `fcn_hidden_dim` | int | Hidden dimension for FCN (default: 1024) |
| `gen.use_adain` | bool | Use AdaIN in the CNN generator |
| `gen.num_downsample` | int | Number of downsampling layers in CNN |

**Training**

| Option | Description |
|---|---|
| `train_iters` | Total training iterations |
| `batch_size` | Batch size for unpaired data |
| `lr` | Learning rate |
| `one_sided` | If True, only train A → B translation |

**Loss weights**

| Option | Description |
|---|---|
| `gen_w` | Generator adversarial loss weight |
| `dis_w` | Discriminator loss weight |
| `recons_w` | Cycle reconstruction loss weight |
| `paired_loss_w` | Anchor / paired supervision loss weight |
| `jacobian_loss_w` | Jacobian regularization weight |
| `r1_reg_w` | R1 gradient penalty weight |

**Jacobian regularization**

| Option | Description |
|---|---|
| `jacobian_loss_w` | Jacobian regularization weight (0 to disable) |
| `jacobian_norm_type` | `fd_l1` (sparse finite difference) or `l1` |
| `jacobian_num_samples` | Number of samples for stochastic estimation |
| `jacobian_probe_sparsity` | Sparsity of perturbation mask (used by `fd_l1`) |
| `jacobian_sigma` | Std. dev. of perturbation magnitude (used by `fd_l1`) |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{shrestha2026domain,
  title     = {Domain Transfer Becomes Identifiable via a Single Alignment},
  author    = {Shrestha, Sagar and Timilsina, Subash and Nguyen, Hoang-Son and Fu, Xiao},
  booktitle = {Proceedings of International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```