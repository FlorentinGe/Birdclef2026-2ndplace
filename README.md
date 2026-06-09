# BirdCLEF+ 2026

Training codebase for the 2nd place solution to Kaggle's [BirdCLEF+ 2026 competition](https://www.kaggle.com/competitions/birdclef-2026/overview) — acoustic species identification from passive monitoring recordings in the Pantanal wetlands, South America.

Kaggle write-up: https://www.kaggle.com/competitions/birdclef-2026/writeups/2nd-place-diverse-ensemble-with-pseudo-labeling-a

**Task**: predict the presence/absence of 234 species in 5-second chunks of 1-minute soundscape recordings.  
**Metric**: macro-averaged ROC-AUC (classes with no true positives in the test set are skipped).

---

## Repository layout

```
Codebase/
├── train.py                        ← main entry point (kd / supervised / focal_pl / sc_pl / ai_specialist)
├── train_distilled_sed.py          ← local adapter for the distilled-SED notebook
├── requirements.txt
│
├── birdclef/                       ← core library
│   ├── config.py                   ← Config dataclass + YAML/CLI loading
│   ├── datasets.py                 ← all Dataset classes
│   ├── model.py                    ← BirdCLEFModel (SED), PretrainModel (KD)
│   ├── transforms.py               ← GPU-side MelTransform + CPU helpers
│   ├── train.py                    ← run_kd_stage, run_supervised_stage
│   ├── validate.py                 ← validate_composite, save_oof_predictions
│   └── utils.py                    ← seeding, vocab, soundscape split, path helpers
│
├── configs/
│   ├── backbone/
│   │   ├── eca_nfnet_l0.yaml
│   │   ├── tf_efficientnet_b3.ns_jft_in1k.yaml
│   │   ├── tf_efficientnet_b4.ns_jft_in1k.yaml
│   │   ├── tf_efficientnetv2_s.in21k_ft_in1k.yaml
│   │   ├── tf_efficientnetv2_m.in21k_ft_in1k.yaml
│   │   ├── convnext_base.fb_in22k_ft_in1k.yaml
│   │   ├── convnext_base.clip_laion2b_augreg_ft_in1k.yaml
│   │   ├── convnextv2_tiny.fcmae_ft_in22k_in1k.yaml
│   │   ├── convnextv2_small.fcmae_ft_in22k_in1k.yaml
│   │   ├── convnextv2_base.fcmae_ft_in22k_in1k.yaml
│   │   ├── regnety_016.tv2_in1k.yaml
│   │   ├── regnety_032.yaml
│   │   ├── efficientvit_l2.r288_in1k.yaml
│   │   ├── passt_base.yaml
│   │   ├── passt_light.yaml
│   │   ├── vit_small_patch16_224.dino.yaml
│   │   └── vit_base_patch16_224.dino.yaml
│   └── stage/
│       ├── kd.yaml
│       ├── supervised.yaml
│       └── sc_pl.yaml               ← focal_pl and ai_specialist have no stage yaml
│
├── generate_sc_pl.py               ← run checkpoints on unlabelled soundscapes → per-model .npy
├── generate_sc_pl_20s.py           ← same with overlapping 20s-window inference
├── generate_focal_pl.py            ← ensemble checkpoints on focal clips → focal_pl_raw_ensemble.npy;
│                                     when --perch_npz is given, also writes
│                                     focal_pl_preds_perch_continuous.npy (z-score soft labels
│                                     consumed by --perch_max at train time)
├── extract_perch_focal.py          ← Perch v2 logits for focal clips via perch-hoplite;
│                                     writes perch_focal_arrays.npz (raw logits, input to
│                                     generate_focal_pl.py --perch_npz)
├── extract_perch_soundscape.py     ← Perch v2 pseudo-labels for soundscapes (overlapping)
├── blend_sc_pl.py                  ← merge per-model sc_pl .npy files into an ensemble array
├── extract_oof_from_checkpoint.py  ← recover OOF predictions from a saved checkpoint
├── export_onnx.py                  ← export a checkpoint to ONNX (5s or 20s input)
└── attention_viz.py                ← SED attention quality diagnostics
```

---

## Setup

```bash
# 1. Install PyTorch with the correct CUDA build for your instance
#   Kepler / Maxwell / Pascal / Volta / Turing — sm_37–sm_75 (K80, P100, T4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#
#   Ampere — sm_80–sm_86 (A100, RTX 30xx):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#
#   Ada Lovelace — sm_89 (RTX 40xx, RTX 4000 Ada/Pro series):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
#
#   Blackwell — sm_120 (RTX 50xx, RTX PRO 4000/5000 Blackwell) — requires driver ≥ 570:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#
#   Using a wheel that doesn't include your GPU's sm_XX causes:
#     "RuntimeError: CUDA error: no kernel image is available for execution on the device"

# 2. Install remaining dependencies
pip install -r requirements.txt
```

---

## Data

All paths default to the standard Kaggle competition mount. The codebase reads **OGG originals** by default (`base_dir/train_audio/`).

| Data | Default path | Size |
|------|-------------|------|
| Competition data | `/kaggle/input/competitions/birdclef-2026/` | ~13 GB |
| Focal clips (OGG) | `…/train_audio/` | ~11 GB |
| Train soundscapes | `…/train_soundscapes/` | included above |
| Perch2 embeddings | `/kaggle/input/birdclef2026-perch-embeddings/` | ~2 GB |
| Saved artefacts | `/kaggle/input/birdclef2026-training-artefacts/` | checkpoint + label map |

To use WAV shards instead of OGG (e.g. if they are already attached):

```bash
python train.py --backbone eca_nfnet_l0 --stage supervised \
    --train_audio_wav_prefix /kaggle/input/datasets/ttahara/birdclef2026-train-audio-wav-
```

---

## Example usage

In order to help with reproducibility, the example commands are similar to those I used for my solution, so they will be more verbose than necessary.

### Stage 1 — Perch2 knowledge distillation (optional warm-start)

Trains the backbone to align its mel-spectrogram embeddings with stored Perch2 embeddings via cosine loss. Saves `pretrained_backbone.pth`.

```bash
python train.py \
    --backbone eca_nfnet_l0 \
    --stage kd \
    --pretrain_epochs 15 \
    --pretrain_batch_size 128 \
    --use_amp true \
    --use_bf16 true --num_workers 8 \
    --perch_embed_dirs \
        /path/to/embeds/dir1 \
        /path/to/embeds/dir2 \
        /path/to/embeds/dir3 \
    --output_dir /path/to/experiment
```

### Stage 2 — Supervised training (+ SWA)

From a pretrained backbone (remove --checkpoint for timm-pretrained):


```bash
python train.py --backbone eca_nfnet_l0 --stage supervised \
    --checkpoint /path/to/pretrained/backbone.pth \
    --freeze_epochs 2 --batch_size 64 --use_amp true --use_bf16 true --num_workers 8 --num_epochs 20 --swa_start_epoch 11 \
    --use_gem true --output_dir /path/to/experiment
```

### Stage 3 — Focal pseudo-labels

Requires a directory with focal_pl.csv (manifest), focal_pl_raw_ensemble.npy (labels) and optionally focal_pl_preds_perch_continuous.npy (Perch labels).

To generate pseudo-labels using existing models:

```bash
python generate_focal_pl.py --model_dirs /path/to/model1 /path/to/model2 --base_dir /workspace/Birdclef/datasets/birdclef-2026 \
    --out_dir /path/to/focal_pl --batch_size 32  --threshold 0.1 --perch_npz /path/to/perch_focal_arrays.npz
```

Raw Perch logits are extracted with `extract_perch_focal.py` (writes `perch_focal_arrays.npz`), then passed to `generate_focal_pl.py --perch_npz` which converts them to continuous z-score soft labels (`focal_pl_preds_perch_continuous.npy`) consumed by `--perch_max` at training time. Perch labels are applied via a class-conditional cap rather than a hard threshold, because Perch reliability varies by taxonomic class.


To train a model from a pretrained backbone (remove --checkpoint for timm-pretrained):

```bash
python train.py --backbone eca_nfnet_l0 --stage focal_pl \
    --batch_size 32 --use_amp true --use_bf16 true --num_workers 8 --num_epochs 20 --swa_start_epoch 11 \
    --focal_pl_csv "/path/to/focal_pl.csv" --output_dir /path/to/experiment \
    --pl_pseudo_th 0.3 --pl_pseudo_alpha 0.7 --pl_pseudo_power 2.0 --pl_perch_max 0.5 --hop_length 512 \
    --checkpoint /path/to/pretrained/backbone.pth
```

### Stage 4 — Soundscape pseudo-label fine-tuning

Implements the 2025 BirdCLEF 1st-place "Multi-Iterative Noisy Student" recipe. Requires a directory with sc_pl.csv (manifest), sc_pl_preds_ensemble.npy (labels) and optionally sc_pl_val_files.txt (list of fixed held-out soundscapes for validation).
 
```bash
# Step 1: generate per-model soundscape pseudo-labels. Use generate_sc_pl_20s.py if trained on 20s windows.
python generate_sc_pl.py --model_dirs /path/to/model1 /path/to/model2 --checkpoints best swa --tta \
    --base_dir /workspace/Birdclef/datasets/birdclef-2026 --soundscape_dir /workspace/Birdclef/datasets/birdclef-2026/train_soundscapes \
    --out_dir /path/to/sc_pl_round1 --batch_size 24 --device cuda

# Step 2: Blend ensemble predictions (with optional gating for models with different calibration):
python blend_sc_pl.py --sc_pl_dir path/to/sc_pl_round1 \
    --model_weights model1:1.0 model2:1.0 model3:1.0 model4:1.0 \
    --gates perch exp54

# Step 3: fine-tune with pseudo-labels (round 1)

For pure BCE training:
python train.py --backbone tf_efficientnetv2_s.in21k_ft_in1k --stage sc_pl --sc_pl_dir /path/to/sc_pl_round1 --batch_size 32 --use_amp true --use_bf16 true \
    --num_workers 8 --num_epochs 30 --swa_start_epoch 16 --output_dir /path/to/experiment --hop_length 512 --base_dir /workspace/Birdclef/datasets/birdclef-2026 \
    --pl_sc_pseudo_power 1.2 --use_llrd false --lr=5e-4 --sc_use_window false --sc_pl_exclude_labelled false --sc_pl_sub_prob 0.5 \
    --duration 5 --time_mask 20 --scheduler cosine --n_mels 128 --fmin 20 --n_fft 2048 --imagenet_norm false \
    --checkpoint /path/to/backbone.pth --use_gem false --use_soft_auc_loss false

For soft AUC + BCE training:
python train.py --backbone tf_efficientnetv2_s.in21k_ft_in1k --stage sc_pl --sc_pl_dir /path/to/sc_pl_round1 --batch_size 32 --use_amp true --use_bf16 true \
    --num_workers 8 --num_epochs 35 --swa_start_epoch 18 --output_dir /path/to/experiment --hop_length 512 --base_dir /workspace/Birdclef/datasets/birdclef-2026 \
    --pl_sc_pseudo_power 1.0 --use_llrd false --lr=5e-4 --sc_use_window false --sc_pl_exclude_labelled false --sc_pl_sub_prob 0.6 \
    --duration 5 --time_mask 20 --scheduler cosine --n_mels 128 --fmin 20 --n_fft 2048 --imagenet_norm false \
    --checkpoint /path/to/backbone.pth --use_gem false --use_soft_auc_loss true --soft_auc_bce_weight 0.25 --sc_pl_hard_labels true
```

### Stage 5 - Insecta/Amphibia specialist

Trains a specialist model on insecta/amphibia data only, using extra XC data. Can optionally use pseudo-labeled soundscapes, although that branch didn't perform well in the competition.
sc_pl_dir can be specified for diagnostic files generation, even if no pseudo-labels are used. To use the pseudo-labels, add the --ai_use_sc_pl true option.

```bash
python train.py --backbone tf_efficientnet_b0.ns_jft_in1k --stage ai_specialist --sc_pl_dir /path/to/sc_pl_round1 --batch_size 128 --use_amp true --use_bf16 true --num_workers 8 \
    --num_epochs 40 --swa_start_epoch 28 --output_dir /path/to/experiment --hop_length 1252 --base_dir /workspace/Birdclef/datasets/birdclef-2026 --use_llrd false --lr=5e-4 --sc_use_window true \
    --duration 20 --time_mask 80 --scheduler cosine --n_mels 224 --fmin 0 --n_fft 4096 --imagenet_norm false --checkpoint /path/to/backbone.pt --use_gem false --use_soft_auc_loss false --sc_use_window true \
    --ai_xc_species_csv /path/to/extra/birdclef2025_extra_species_data.csv --ai_xc_species_dir /path/to/extra/data/
```

### 'Distilled SED'-inspired training

Standalone script forked from Tucker Arrants' public notebook. Supports optional
Perch distillation using cached embeddings, controlled by the `BACKBONE_MODE` constant at the top of the file:

| Mode | Behaviour |
|------|-----------|
| `distill_sg` | Perch distillation with stop-gradient on the classification head |
| `distill_nosg` | Perch distillation, no stop-gradient (backbone sees both losses) |
| `no_distill` | Classification only — distillation removed (used in the competition) |
| `frozen` | Frozen backbone, classification head only |

Loss function is selectable: pure BCE or soft AUC + BCE.


### Run-level overrides

Any parameter in `Config` can be overridden via a YAML file:

```yaml
# configs/local_paths.yaml
base_dir: /data/birdclef-2026
output_dir: ./runs/exp_local
num_workers: 8
```

```bash
python train.py --backbone eca_nfnet_l0 --stage supervised \
    --config configs/local_paths.yaml
```

Config loading order (later entries win):

1. `Config` dataclass defaults
2. `configs/backbone/<backbone>.yaml`
3. `configs/stage/<stage>.yaml`
4. `--config <path>.yaml` (optional)
5. Explicit CLI flags

---

## Model architecture

**SED (Sound Event Detection) head** — used in all experiments:

```
timm backbone (global_pool='') → (B, C, H, W) feature map
  → mean over freq axis H      → (B, C, T)
  → BatchNorm1d + Dropout(0.3)
  → fc(C→num_classes)          → frame_logits  (B, T, classes)
  → att_fc(C→num_classes)      → att_logits    (B, T, classes)

clipwise_logits = frame_logits.mean(dim=1)                      ← training loss
att_clipwise    = (frame_logits × softmax(att_logits)).sum(1)   ← val / inference
```

The ViT path (any `vit_*` backbone) reshapes the flat patch token sequence into the `(H_patches, W_patches)` spatial grid before applying the same SED head.

---

## Training pipeline

| Stage | Script arg | What it does |
|-------|-----------|--------------|
| 1 — KD | `--stage kd` | Cosine distillation from Perch2 embeddings; saves backbone weights |
| 2 — Supervised | `--stage supervised` | BCE classification on focal clips + labelled soundscape chunks; SWA from `swa_start_epoch` |
| 3 — Focal PL | `--stage focal_pl` | BCE on focal clips with CNN ensemble soft labels (power-transform + alpha blend); optional Perch discovery layer via --perch_max |
| 4 — SC PL | `--stage sc_pl` | Noisy-student fine-tuning with soundscape pseudo-labels; power transform + weighted sampler |
| 5 — AI specialist | `--stage ai_specialist` | Amphibia/Insecta-only training with extra XC data |

All stages use:
- **Waveform-domain mixup** (λ = 0.5, union labels) before GPU mel conversion
- **SpecAugment** (FrequencyMasking + TimeMasking) on the mel spectrogram
- **AdamW** + cosine scheduler with linear warmup
- **Gradient clipping** after `scaler.unscale_()` (critical for correct AMP behaviour)
- **SWA** — BN running stats copied directly from the last training-epoch model rather than re-estimated via `update_bn()` (which causes NaN stats with NFNet/EfficientNetV2 SED heads)
---

## Validation

Two-part **composite AUC**:

```
composite = (sc_mean × n_sc + focal_mean × n_focal) / (n_sc + n_focal)
```

- **Soundscape AUC** (~15 classes): 15% of labelled soundscape files — domain-matched to the test set
- **Focal AUC** (~188 classes): 10% of focal clips - can have some overlap with the above
- **Unlabeled soundscapes**: 10% of the set are held out, used for cosine similarity metrics during training and oof comparisons.
- Per-class AUC computed with a per-class loop (not `sklearn average='macro'`) to correctly skip all-zero columns
- At least 1 soundscape file per soundscape-only species is forced into training. As a result, some species are not covered in the validation set (most notably the insecta sonotypes).
- The validation set is kept constant throughout all training stages

---

## Outputs

Each training run writes to `output_dir/`:

| File | Description |
|------|-------------|
| `best_model.pth` | Best per-epoch model weights |
| `swa_model.pth` | SWA-averaged model weights (primary submission artefact) |
| `label_map.npy` | `{class_index: species_code}` dict |
| `epoch_history.csv` | Per-epoch loss / AUC / LR / timing |
| `per_class_auc.csv` | Per-class AUC at every epoch (soundscape + focal splits) |
| `oof_swa_{sc,focal}.npz` | OOF predictions from SWA model |
| `oof_best_{sc,focal}.npz` | OOF predictions from best-epoch model |
| `run_summary.json` | Best AUC, SWA AUC, config snapshot |
| `run_config.json` | Full serialised Config |
| `pretrained_backbone.pth` | KD stage only: backbone weights for Stage 2 warm-start |
| `kd_history.csv` | KD stage only: per-epoch cosine similarity |

---

## Inference

The inference notebook (`notebooks/example single model inference script.py`) is CPU-only with a 90-minute hard limit for 600 test files. It uses:
- `torchaudio` for vectorised mel conversion (no per-chunk Python loop)
- All 12 chunks of a 1-minute file batched in a single forward pass
- `att_clipwise` output (attention-weighted probabilities)
- TTA: 3-pass temporal roll (±1.25s), optional
- Confidence scaling post-processing, optional

The `BirdCLEFModel` is **incompatible with TorchScript** (dynamic feature-size detection in `__init__`). Use `model.eval()` + `torch.inference_mode()` instead.

`notebooks/2nd place ensemble inference script.py` contains the 2nd-place submission ensemble.
