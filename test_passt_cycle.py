"""
End-to-end sanity check for the PaSST training cycle.

Runs with the real hear21passt installed.  Does NOT need a GPU or real data —
it uses random tensors on CPU/CUDA (whichever is available).

Run from the Codebase directory:
    python test_passt_cycle.py

Checks:
  1. Model builds without error (including dummy forward during __init__)
  2. Forward pass shapes for 10s train input and 5s val input
  3. Full mini-batch training step: forward → loss → backward → optimizer step
  4. Gradient flows to both backbone and SED head
  5. BF16 autocast forward pass (training-time AMP)
  6. PretrainModel (KD) forward + backward
  7. GeM variant
"""

import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '.')

# ── Config ────────────────────────────────────────────────────────────────────
from birdclef.config import Config
from birdclef.model import BirdCLEFModel, PretrainModel
from birdclef.transforms import MelTransform

def make_cfg():
    cfg = Config()
    cfg.backbone        = 'passt_base'
    cfg.in_chans        = 1
    cfg.n_mels          = 128
    cfg.n_fft           = 1024
    cfg.hop_length      = 320
    cfg.fmin            = 0
    cfg.fmax            = 16000
    cfg.imagenet_norm   = False
    cfg.use_gem         = False
    cfg.gem_p_init      = 3.0
    cfg.use_llrd        = False
    cfg.drop_path_rate  = 0.0
    cfg.duration        = 10       # → img_size (128, 1001)
    cfg.chunk_duration  = 5        # → img_size_chunk (128, 501)
    cfg.sr              = 32000
    cfg.perch_emb_dim   = 1536
    cfg.proj_mlp        = False
    cfg.freq_mixstyle   = False
    cfg.freq_mask       = 30
    cfg.time_mask       = 80
    cfg.use_amp         = False    # test in FP32 first; BF16 tested separately
    cfg.use_bf16        = False
    return cfg

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 206
BATCH       = 2   # small batch to keep the test fast

print(f"Device: {DEVICE}")
print(f"hear21passt installed: ", end='')
try:
    from hear21passt.base import get_basic_model
    print("YES")
except ImportError:
    print("NO — test requires hear21passt to be installed")
    sys.exit(1)

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(cond, msg):
    tag = PASS if cond else FAIL
    print(f"  {tag}  {msg}")
    if not cond:
        raise AssertionError(msg)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. Build BirdCLEFModel (PaSST, no GeM)")

cfg = make_cfg()
model = BirdCLEFModel(cfg, num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
mel_xf = MelTransform(cfg).to(DEVICE)

# ── 10s training clip (waveform → mel → model) ────────────────────────────────
SR      = cfg.sr
wave10  = torch.randn(BATCH, SR * 10).to(DEVICE)          # 10s waveform
with torch.no_grad():
    mel10 = mel_xf(wave10, augment=False)                  # (B, 1, 128, 1001)

print(f"  mel10 shape : {list(mel10.shape)}", end='  ')
check(list(mel10.shape) == [BATCH, 1, 128, 1001], f"expected [{BATCH},1,128,1001]")

model.eval()
with torch.no_grad():
    clip_logits, att_clip, frame_logits = model(mel10)

check(list(clip_logits.shape)  == [BATCH, NUM_CLASSES], "clipwise_logits shape")
check(list(att_clip.shape)     == [BATCH, NUM_CLASSES], "att_clipwise shape")
check(frame_logits.shape[0]    == BATCH,                "frame_logits batch dim")
check(frame_logits.shape[2]    == NUM_CLASSES,          "frame_logits class dim")
T_train = frame_logits.shape[1]
print(f"  frame_logits T_train = {T_train}  (expected 99 for 10s @ stride=10)")
check(T_train == 99, f"W_time for 10s = 99")

# ── 5s val chunk ──────────────────────────────────────────────────────────────
wave5 = torch.randn(BATCH, SR * 5).to(DEVICE)
with torch.no_grad():
    mel5 = mel_xf(wave5, augment=False)                    # (B, 1, 128, 501)

print(f"  mel5 shape  : {list(mel5.shape)}", end='  ')
check(list(mel5.shape) == [BATCH, 1, 128, 501], f"expected [{BATCH},1,128,501]")

with torch.no_grad():
    _, att_v, fl_v = model(mel5)
check(list(att_v.shape) == [BATCH, NUM_CLASSES], "val att_clipwise shape")
T_val = fl_v.shape[1]
print(f"  frame_logits T_val = {T_val}  (expected 49 for 5s @ stride=10)")
check(T_val == 49, "W_time for 5s = 49")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. Full mini-batch training step (FP32)")

model.train()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

labels = torch.zeros(BATCH, NUM_CLASSES, device=DEVICE)
labels[:, 0] = 1.0   # dummy positive

optimizer.zero_grad()
specs = mel_xf(wave10, augment=True)
clip_logits, _, frame_logits = model(specs)
loss = criterion(clip_logits, labels)
loss.backward()
optimizer.step()

check(math.isfinite(loss.item()), f"loss is finite: {loss.item():.4f}")

# Check gradients flow to both backbone and SED head
backbone_grad = any(
    p.grad is not None and p.grad.abs().sum().item() > 0
    for p in model.backbone.parameters() if p.requires_grad
)
head_grad = any(
    p.grad is not None and p.grad.abs().sum().item() > 0
    for p in model.fc.parameters()
)
check(backbone_grad, "gradients flow to PaSST backbone")
check(head_grad,     "gradients flow to SED head (fc)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. BF16 autocast forward (training-time AMP)")

if DEVICE.type == 'cuda':
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        clip_bf16, att_bf16, _ = model(specs)
    check(list(clip_bf16.shape) == [BATCH, NUM_CLASSES], "BF16 forward shape")
    print("  BF16 autocast OK")
else:
    print("  SKIP (CPU — BF16 autocast not tested)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. BirdCLEFModel with GeM")

cfg_gem = make_cfg()
cfg_gem.use_gem = True
model_gem = BirdCLEFModel(cfg_gem, num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
model_gem.eval()
with torch.no_grad():
    _, att_g, _ = model_gem(mel10)
check(list(att_g.shape) == [BATCH, NUM_CLASSES], "GeM att_clipwise shape")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. PretrainModel (KD) — forward + backward")

cfg_kd = make_cfg()
pretrain = PretrainModel(cfg_kd, pretrained=True).to(DEVICE)

# KD uses 5s chunks (chunk_duration)
wave5_kd = torch.randn(BATCH, SR * 5).to(DEVICE)
mel_kd   = MelTransform(cfg_kd).to(DEVICE)
with torch.no_grad():
    specs5 = mel_kd(wave5_kd, augment=False)   # (B, 1, 128, 501)

check(list(specs5.shape) == [BATCH, 1, 128, 501], "KD mel shape")

opt_kd = optim.AdamW(pretrain.parameters(), lr=1e-4)
target = torch.randn(BATCH, 1536, device=DEVICE)
target = target / target.norm(dim=1, keepdim=True)   # L2-normalise

opt_kd.zero_grad()
emb = pretrain(specs5)
kd_loss = 1.0 - (emb * target).sum(dim=1).mean()
kd_loss.backward()
opt_kd.step()

check(list(emb.shape) == [BATCH, 1536],    "KD embedding shape")
check(math.isfinite(kd_loss.item()),        f"KD loss finite: {kd_loss.item():.4f}")
norms = emb.detach().norm(dim=1)
check(all(abs(n - 1.0) < 1e-4 for n in norms.tolist()),
      f"KD embeddings L2-normalised: {norms.tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("6. _CaptureNorm is in place (no hooks on norm layer)")

# Verify the norm was replaced with _CaptureNorm
from birdclef.model import _CaptureNorm
check(isinstance(model.backbone._net.norm, _CaptureNorm),
      "backbone._net.norm is _CaptureNorm (not plain LayerNorm)")
check(not model.backbone._net.norm._forward_pre_hooks,
      "no forward_pre_hooks on norm layer (hooks removed)")

print("\n" + "="*60)
print("7. SWA deepcopy — forward must work on the averaged model")
# Regression test for the lambda-closure bug:
#   AveragedModel deep-copies the backbone, but Python lambdas are atomic
#   in deepcopy.  The hook must write to the norm module (`m`), not to `self`
#   via a closure, otherwise the SWA copy's forward crashes with AttributeError.

from torch.optim.swa_utils import AveragedModel

swa_model = AveragedModel(model)
swa_model.update_parameters(model)   # copy weights into the averaged model

swa_model.eval()
with torch.no_grad():
    try:
        _, att_swa, _ = swa_model(mel10)
        check(list(att_swa.shape) == [BATCH, NUM_CLASSES],
              "SWA deep-copy forward shape")
        print("  SWA forward OK (deepcopy hook fix confirmed)")
    except AttributeError as e:
        check(False, f"SWA forward failed — deepcopy/hook bug still present: {e}")

# Also check 5s val chunk through SWA model
with torch.no_grad():
    _, att_swa5, _ = swa_model(mel5)
check(list(att_swa5.shape) == [BATCH, NUM_CLASSES], "SWA 5s forward shape")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL CHECKS PASSED")
