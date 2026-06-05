"""
Model definitions for BirdCLEF+ 2026.

BirdCLEFModel  — SED architecture, handles both CNN and ViT backbones.
PretrainModel  — Stage 1 KD: backbone + projection head → L2-normalised embedding.

Both CNN and ViT forward pass return (clipwise_logits, att_clipwise, frame_logits).
  clipwise_logits : (B, num_classes) — simple mean over time; use for training loss
  att_clipwise    : (B, num_classes) — attention-weighted; use for val / inference
  frame_logits    : (B, T, num_classes) — per-frame; returned for completeness

When forward() is called with return_att=True, a 4th tensor is appended:
  att_weights     : (B, T, num_classes) — softmax over T; use for attention diagnostics

ViT detection: any timm model starting with 'vit_' or 'deit_'.  All other
backbones are treated as CNNs (return a (B, C, H, W) feature map).

Important:
  The dummy forward pass in __init__ detects the backbone output shape
  dynamically.  This makes the model INCOMPATIBLE with TorchScript.
  Use model.eval() + torch.inference_mode() at inference time instead.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


# ── GeM pooling ───────────────────────────────────────────────────────────────

class GeMPool1d(nn.Module):
    """
    Generalised Mean Pooling over the frequency axis (dim=2) of a 4-D feature map.

    Replaces feat.mean(dim=2) in both CNN and ViT SED paths.
    p=1 recovers average pooling; p→∞ approaches max pooling.
    p is a learnable scalar initialised to p_init (default 3.0).

    Negative features are clamped to eps before raising to the power p,
    which is consistent with standard GeM usage (backbone outputs are
    expected to be non-negative after activations in most architectures;
    for NFNet the clamp has negligible impact in practice).
    """

    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — pool over H (freq axis dim=2) → (B, C, W)
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


# ── Helper: backbone-family detection ────────────────────────────────────────

_VIT_PREFIXES = ('vit_', 'deit_', 'beit_', 'eva_')


def _is_vit(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _VIT_PREFIXES)


def _is_passt(model_name: str) -> bool:
    return model_name.startswith('passt')


# Maps our backbone name → hear21passt arch string.
# All variants use ViT-B (embed_dim=768); the "s" in hear21passt means
# "structured patchout" (not "small").  passt_light is the only genuinely
# lighter model: depth=7 instead of 12, same stride=10 and D=768.
_PASST_ARCH = {
    'passt_base':  'passt_s_swa_p16_128_ap476',   # depth=12, SWA pretrained
    'passt_light': 'passt_l_kd_p16_128_ap47',      # depth=7,  KD  pretrained
}


# ── PaSST feature-capture norm wrapper ───────────────────────────────────────

class _CaptureNorm(nn.Module):
    """
    Drop-in replacement for the PaSST final LayerNorm that also stores its
    **pre-norm** input so ``PaSSTBackbone.forward`` can extract patch features.

    Why not a forward pre-hook?
    ---------------------------
    The original implementation used ``register_forward_pre_hook`` with a lambda
    that closed over the backbone instance via ``self``.  Python treats all
    function objects (including lambdas) as atomic in ``deepcopy`` — the closure
    always references the *original* backbone, so the SWA ``AveragedModel`` deep
    copy never got the captured tensor and crashed with ``AttributeError: 'NoneType'
    object has no attribute 'dim'``.

    Storing on the norm *module* (``m._passt_captured``) fixed the deepcopy bug
    but the approach still uses a hook, which breaks ``torch.onnx.export``:
    hooks fire as Python side effects during tracing; the ONNX tracer cannot
    guarantee the data-flow edge is preserved in the exported graph.

    ``_CaptureNorm`` replaces the hook entirely.  ``self._captured = x`` inside
    ``forward`` is a plain tensor assignment on a proper ``nn.Module``; the ONNX
    tracer sees it as a live traced tensor and correctly records the edge
    ``last_transformer_block → _captured → SED_head``.  deepcopy also works
    correctly because modules are fully copied.
    """

    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.norm     = norm
        self._captured: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._captured = x          # store pre-norm tokens (B, N+2, D)
        return self.norm(x)         # return the usual normalised output


# ── PaSST backbone wrapper ────────────────────────────────────────────────────

class PaSSTBackbone(nn.Module):
    """
    Wraps a hear21passt PaSST model to expose a CNN-compatible 4-D feature map
    for our SED head.

    Supported backbones (see ``_PASST_ARCH``)
    ------------------------------------------
    passt_base  — ``passt_s_swa_p16_128_ap476``: depth=12, stride=10, SWA.
                  The "s" = structured patchout (NOT "small").
    passt_light — ``passt_l_kd_p16_128_ap47``: depth=7,  stride=10, KD.
                  "L" = Light.  Same D=768, ~58 % of passt_base transformer cost.

    Feature map layout
    ------------------
    hear21passt's DeiT-style ``forward_features`` internally processes tokens as
    ``(B, N+2, D)`` (CLS + dist + N patch tokens) and returns ``(cls, dist)``
    — both ``(B, D)``.  ``_CaptureNorm`` replaces the final LayerNorm and stores
    the pre-norm token sequence so we can extract it, drop CLS and dist, and
    reshape into a CNN-compatible 4-D feature map:

        (B, N, D) → (B, H_freq, W_time, D) → (B, D, H_freq, W_time)

    where ``H_freq = patch_embed.grid_size[0]`` (read from the model, not
    hardcoded).  The result is consumed by the existing CNN SED path.

    Positional embedding constraint
    --------------------------------
    Both pretrained models used 10-second AudioSet clips → ``W_time = 99`` at
    stride=10.  Training clips must therefore use ``duration: 10`` so the patch
    grid matches exactly.  Validation chunks (5 s → W_time = 49 ≤ 99) are
    handled by PaSST's CUT truncation.

    Patchout disabled — our SpecAugment provides equivalent regularisation.
    """

    def __init__(self, arch: str = 'passt_s_swa_p16_128_ap476', pretrained: bool = True):
        super().__init__()
        try:
            from hear21passt.models.passt import get_model as _get_passt_net
        except ImportError:
            raise ImportError(
                'hear21passt is required for PaSST backbone.\n'
                "Install with: pip install 'hear21passt>=0.0.19'"
            )

        # hear21passt.base.get_basic_model is NOT used here because it
        # hardcodes pretrained=True internally, always downloading AudioSet
        # weights regardless of our pretrained flag.  We call get_model()
        # directly so pretrained=False skips the download (e.g. when loading
        # a checkpoint or running offline inference on Kaggle).
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):   # suppress print(model)
            self._net = _get_passt_net(
                arch=arch,
                pretrained=pretrained,
                n_classes=527,
                in_channels=1,
                fstride=10,
                tstride=10,
                input_fdim=128,
                input_tdim=998,
            )

        # Disable patchout so the feature map shape stays regular.
        for attr in ('u_patchout', 's_patchout_t', 's_patchout_f'):
            if hasattr(self._net, attr):
                setattr(self._net, attr, 0)

        # Expose transformer blocks for LLRD compatibility.
        self.blocks = self._net.blocks

        # Read H_freq from the model's patch_embed so this works for any stride.
        self._h_patches: int = self._net.patch_embed.grid_size[0]

        # Replace the final LayerNorm with a _CaptureNorm wrapper.
        # This is the canonical way to extract pre-norm features:
        #   • No forward hooks (hooks are Python side effects invisible to ONNX)
        #   • deepcopy-safe (proper nn.Module; no lambda closures)
        #   • ONNX-traceable (plain tensor assignment inside forward)
        self._net.norm = _CaptureNorm(self._net.norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, 1, 128, T)`` mel spectrogram.

        Returns:
            ``(B, 768, H_freq, W_time)`` 4-D feature map, consumed by the CNN
            SED path in BirdCLEFModel.

        The ``FMAX is None`` warning printed by hear21passt is benign:
        it initialises an internal mel transform we never invoke — we pass
        our own mel spectrograms directly.
        """
        self._net.forward_features(x)     # _CaptureNorm stores pre-norm tokens
        feat = self._net.norm._captured   # (B, N+2, D)

        if feat.dim() == 4:
            # Some hear21passt builds reshape tokens before norm; use directly.
            return feat

        # 3D path: (B, N+2, D) — drop CLS (0) and dist (1), keep patch tokens.
        patch = feat[:, 2:, :]                # (B, N, D)
        B, N, D = patch.shape
        W_p = N // self._h_patches
        # Reshape row-major (freq-first) then move D to channel dim.
        return (patch
                .reshape(B, self._h_patches, W_p, D)
                .permute(0, 3, 1, 2))         # (B, D, H_freq, W_time)


# ── Stem stride patch ─────────────────────────────────────────────────────────

def _patch_stem_stride(backbone: nn.Module, backbone_name: str, stride: int) -> None:
    """Patch the first stride-2 stem conv to a different stride value.

    Call after timm.create_model() and before the dummy forward pass so that
    the dynamic feature-size detection in BirdCLEFModel / PretrainModel picks
    up the modified spatial dimensions automatically.

    Supported families:
      EfficientNet / EfficientNetV2  → backbone.conv_stem
      ECA-NFNet (4-conv stem)        → backbone.stem.conv4  (last stride-2 conv)
      RegNetY / RegNetX              → backbone.stem.conv
    ViT and PaSST are silently skipped (their spatial resolution is governed by
    patch size, not a strided conv).
    """
    if stride == 2:
        return  # default — no patch needed

    s = (stride, stride)

    if hasattr(backbone, 'conv_stem'):               # EfficientNet / V2
        backbone.conv_stem.stride = s
        print(f'  stem_stride=1: patched conv_stem.stride → {s}')
    elif hasattr(backbone, 'stem') and hasattr(backbone.stem, 'conv4'):  # NFNet
        backbone.stem.conv4.stride = s
        print(f'  stem_stride=1: patched stem.conv4.stride → {s}')
    elif hasattr(backbone, 'stem') and hasattr(backbone.stem, 'conv'):   # RegNet
        backbone.stem.conv.stride = s
        print(f'  stem_stride=1: patched stem.conv.stride → {s}')
    else:
        print(f'  stem_stride=1: WARNING — {backbone_name!r} has no recognised stem conv; '
              f'stem_stride={stride} has no effect')


# ── Main SED model ────────────────────────────────────────────────────────────

class BirdCLEFModel(nn.Module):
    """
    SED (Sound Event Detection) architecture.

    CNN path (e.g. ECA-NFNet-L0, EfficientNetV2-S, RegNetY-032):
      backbone(x) → (B, C, H, W) feature map
      → mean over freq axis H → (B, C, T)
      → BN + Dropout
      → fc / att_fc per time step
      → att_clipwise = attention-weighted sum

    ViT path (e.g. vit_base_patch16_224.dino):
      backbone(x) → (B, 1+N, D) token sequence  [global_pool='']
      → drop CLS token → (B, N, D)
      → reshape to 2D grid (B, D, H_p, W_p)
      → mean over freq patches H_p → (B, D, W_p)
      → shared SED head (identical to CNN path from here)
    """

    def __init__(self, cfg: Config, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self._use_gem    = cfg.use_gem
        if cfg.use_gem:
            self.gem_pool = GeMPool1d(p_init=cfg.gem_p_init)

        if _is_passt(cfg.backbone):
            # ── PaSST path ──────────────────────────────────────────────────
            # forward_features returns (B, D, H_freq, W_time) via the norm
            # pre-hook + reshape in PaSSTBackbone.  Route through the CNN SED
            # path (mean/GeM over freq dim=2 → (B, D, W_time)).
            self._arch    = 'cnn'
            _passt_arch   = _PASST_ARCH.get(cfg.backbone, cfg.backbone)
            self.backbone = PaSSTBackbone(arch=_passt_arch, pretrained=pretrained)
            with torch.no_grad():
                dummy      = torch.zeros(1, cfg.in_chans, cfg.n_mels, cfg.img_size[1])
                feat       = self.backbone(dummy)   # (1, D, H_freq, W_time)
                n_features = feat.shape[1]          # D = 768

        elif _is_vit(cfg.backbone):
            # ── timm ViT path ────────────────────────────────────────────────
            self._arch    = 'vit'
            self.backbone = timm.create_model(
                cfg.backbone,
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
                in_chans=cfg.in_chans,
                img_size=cfg.img_size,
                drop_path_rate=cfg.drop_path_rate,
                dynamic_img_size=True,
            )
            with torch.no_grad():
                dummy  = torch.zeros(1, cfg.in_chans, cfg.n_mels, cfg.img_size[1])
                tokens = self.backbone(dummy)                  # (1, 1+N, D)
                n_features = tokens.shape[2]
                patch_size = self.backbone.patch_embed.patch_size[0]
                self.h_patches = cfg.n_mels // patch_size

        else:
            # ── timm CNN path ────────────────────────────────────────────────
            self._arch    = 'cnn'
            self.backbone = timm.create_model(
                cfg.backbone,
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
                in_chans=cfg.in_chans,
                drop_path_rate=cfg.drop_path_rate,
            )
            _patch_stem_stride(self.backbone, cfg.backbone, cfg.stem_stride)
            with torch.no_grad():
                dummy      = torch.zeros(1, cfg.in_chans, cfg.n_mels, cfg.img_size[1])
                feat       = self.backbone(dummy)
                n_features = feat.shape[1]

        self.fc      = nn.Linear(n_features, num_classes)
        self.att_fc  = nn.Linear(n_features, num_classes)
        self.bn      = nn.BatchNorm1d(n_features)
        self.dropout = nn.Dropout(0.3)

    def forward(
        self, x: torch.Tensor, return_att: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        if self._arch == 'vit':
            # timm ViT: backbone returns (B, 1+N, D) token sequence
            tokens = self.backbone(x)                         # (B, 1+N, D)
            tokens = tokens[:, 1:, :]                         # drop CLS → (B, N, D)
            B, N, D = tokens.shape
            w_patches = N // self.h_patches
            # reshape flat tokens to 2D patch grid → (B, D, H_p, W_p)
            feat = tokens.reshape(B, self.h_patches, w_patches, D)
            feat = feat.permute(0, 3, 1, 2)                   # (B, D, H_p, W_p)
            feat = self.gem_pool(feat) if self._use_gem else feat.mean(dim=2)  # (B, D, W_p)
            feat = feat.permute(0, 2, 1)                      # (B, T, D)
        else:
            feat = self.backbone(x)                           # (B, C, H, W)
            feat = self.gem_pool(feat) if self._use_gem else feat.mean(dim=2)  # (B, C, W)
            feat = feat.permute(0, 2, 1)                      # (B, T, C)

        B, T, C = feat.shape
        feat = self.bn(feat.reshape(B * T, C)).reshape(B, T, C)
        feat = self.dropout(feat)

        frame_logits    = self.fc(feat)                       # (B, T, classes)
        att_logits      = self.att_fc(feat)
        clipwise_logits = frame_logits.mean(dim=1)            # (B, classes)
        att_weights     = torch.softmax(att_logits, dim=1)    # (B, T, classes)
        att_clipwise    = (frame_logits * att_weights).sum(dim=1)

        if return_att:
            return clipwise_logits, att_clipwise, frame_logits, att_weights
        return clipwise_logits, att_clipwise, frame_logits


# ── KD pre-training model ─────────────────────────────────────────────────────

class PretrainModel(nn.Module):
    """
    Stage 1: backbone + linear projection head for Perch2 embedding distillation.

    CNN path: global average pool over (B, C, H, W) → (B, C) → proj → (B, 1536).
    ViT path: CLS token from (B, 1+N, D) → (B, D) → proj → (B, 1536).

    For CNN: backbone uses global_pool='' + manual mean([2,3]), identical to
    BirdCLEFModel's convention.  Backbone weights transfer to BirdCLEFModel Stage 2
    with no surgery — the projection head is discarded.

    For ViT: DINO specifically trains the CLS token as a global image summary.
    Using CLS here (not patch mean) is canonical for DINO distillation.
    BirdCLEFModel drops the CLS token for its SED patch-based head — the two
    branches use different tokens from the same backbone.

    The projection head has no bias: L2 normalisation after a bias-free linear
    layer is equivalent to cosine similarity in the weight space.
    """

    def __init__(self, cfg: Config, pretrained: bool = True):
        super().__init__()

        if _is_passt(cfg.backbone):
            # ── PaSST path ──────────────────────────────────────────────────
            # forward_features returns (B, D, H_freq, W_time); global avg pool
            # over spatial dims gives a (B, D) clip embedding for KD projection.
            self._arch    = 'passt'
            _passt_arch   = _PASST_ARCH.get(cfg.backbone, cfg.backbone)
            self.backbone = PaSSTBackbone(arch=_passt_arch, pretrained=pretrained)
            with torch.no_grad():
                dummy  = torch.zeros(1, cfg.in_chans, cfg.n_mels, cfg.img_size_chunk[1])
                n_feat = self.backbone(dummy).shape[1]   # D = 768

        elif _is_vit(cfg.backbone):
            # ── timm ViT path ────────────────────────────────────────────────
            self._arch    = 'vit'
            self.backbone = timm.create_model(
                cfg.backbone,
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
                in_chans=cfg.in_chans,
                img_size=cfg.img_size,
                drop_path_rate=cfg.drop_path_rate,
                dynamic_img_size=True,
            )
            with torch.no_grad():
                dummy  = torch.zeros(1, cfg.in_chans, cfg.n_mels, cfg.img_size_chunk[1])
                tokens = self.backbone(dummy)                  # (1, 1+N, D)
                n_feat = tokens.shape[2]                       # D

        else:
            # ── timm CNN path ────────────────────────────────────────────────
            self._arch    = 'cnn'
            self.backbone = timm.create_model(
                cfg.backbone,
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
                in_chans=cfg.in_chans,
                drop_path_rate=cfg.drop_path_rate,
            )
            _patch_stem_stride(self.backbone, cfg.backbone, cfg.stem_stride)
            with torch.no_grad():
                dummy  = torch.zeros(1, cfg.in_chans, cfg.n_mels, cfg.img_size_chunk[1])
                n_feat = self.backbone(dummy).shape[1]

        if cfg.proj_mlp:
            # 2-layer MLP: backbone → hidden (n_feat) → BN → ReLU → perch_emb_dim
            # Gives the backbone more representational freedom — it doesn't need to
            # directly encode Perch2's feature space; the MLP handles the alignment.
            # The linear layer is preferred only when strict weight-space cosine
            # alignment is desired (set proj_mlp=False to reproduce exp19/exp24).
            self.proj = nn.Sequential(
                nn.Linear(n_feat, n_feat, bias=False),
                nn.BatchNorm1d(n_feat),
                nn.ReLU(),
                nn.Linear(n_feat, cfg.perch_emb_dim, bias=False),
            )
        else:
            self.proj = nn.Linear(n_feat, cfg.perch_emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._arch == 'passt':
            feat = self.backbone(x)           # (B, D, H_freq, W_time)
            feat = feat.mean(dim=[2, 3])      # global avg pool over spatial dims → (B, D)
        elif self._arch == 'vit':
            tokens = self.backbone(x)         # (B, 1+N, D)
            feat   = tokens[:, 0, :]          # CLS token → (B, D)
        else:
            feat = self.backbone(x)           # (B, C, H, W)
            feat = feat.mean(dim=[2, 3])      # global avg pool → (B, C)

        proj = self.proj(feat)                # (B, perch_emb_dim)
        return F.normalize(proj, dim=1)       # L2 unit vector


# ── ViT LLRD optimiser builder ────────────────────────────────────────────────

def build_optimizer_with_llrd(
    model: BirdCLEFModel,
    cfg: Config,
) -> List[Dict]:
    """
    Build AdamW parameter groups with layer-wise LR decay (LLRD).

    Supports ViT and ConvNext backbones:

    ViT (e.g. vit_small_patch16_224.dino), layer assignment:
      Layer  0 : SED head (fc, att_fc, bn, dropout)  → base_lr × decay⁰  = base_lr
      Layer  1 : backbone final norm + blocks[N-1]   → base_lr × decay¹
      ⋮
      Layer  N : blocks[0]                            → base_lr × decay^N
      Layer N+1: patch_embed, cls_token, pos_embed   → base_lr × decay^(N+1)

    ConvNext (e.g. convnext_base.clip_laion2b_augreg_ft_in1k), layer assignment:
      Layer  0 : SED head                            → base_lr × decay⁰  = base_lr
      Layer  1 : backbone.head + stages[S-1]         → base_lr × decay¹
      ⋮
      Layer  S : stages[0]                           → base_lr × decay^S
      Layer S+1: stem                                → base_lr × decay^(S+1)

    No weight decay on 1-D params (biases, LayerNorm / BN scale+shift) or on
    positional / class embedding tensors (ndim ≥ 2 but semantically not weights).
    """
    base_lr = cfg.lr
    decay   = cfg.llrd_decay
    wd      = cfg.weight_decay

    is_convnext = hasattr(model.backbone, 'stages')
    is_vit      = hasattr(model.backbone, 'blocks')

    if not is_convnext and not is_vit:
        raise ValueError(
            f'build_optimizer_with_llrd requires a backbone with .blocks (ViT/PaSST) '
            f'or .stages (ConvNext). Backbone {cfg.backbone!r} has neither.'
        )

    if is_convnext:
        n_stages = len(model.backbone.stages)

        def get_layer_id(name: str) -> int:
            if not name.startswith('backbone.'):
                return 0                                 # SED head: full LR
            rest = name[len('backbone.'):]
            if rest.startswith('stem.'):
                return n_stages + 1                      # deepest
            if rest.startswith('stages.'):
                stage_idx = int(rest.split('.')[1])
                return n_stages - stage_idx              # stages[S-1]→1 … stages[0]→S
            return 1                                     # backbone.head (LayerNorm2d)

        depth_label = f'{n_stages} stages'

    elif hasattr(model.backbone, 'blocks'):  # EfficientNet / MobileNet
        # .blocks is a Sequential of 7 MB-Conv stages.
        # conv_head (final 1×1 projection → 1280-d) sits after blocks[-1] and must
        # share its LR, not be lumped with the stem.  The ViT fallback would assign
        # it LID = n_blocks+1 (same as conv_stem), which is incorrect.
        n_blocks = len(model.backbone.blocks)

        def get_layer_id(name: str) -> int:
            if not name.startswith('backbone.'):
                return 0                                 # SED head: full LR
            rest = name[len('backbone.'):]
            if rest.startswith('blocks.'):
                block_idx = int(rest.split('.')[1])
                return n_blocks - block_idx              # blocks[n-1]→1, blocks[0]→n
            if any(rest.startswith(p) for p in ('conv_head', 'bn2', 'act2')):
                return 1                                 # post-blocks projection = last stage
            return n_blocks + 1                          # conv_stem, bn1, act1

        depth_label = f'{n_blocks} EfficientNet stages'

    else:  # ViT / PaSST
        n_blocks = len(model.backbone.blocks)

        def get_layer_id(name: str) -> int:
            if not name.startswith('backbone.'):
                return 0                                 # SED head: full LR
            rest = name[len('backbone.'):]
            if rest.startswith('norm.'):
                return 1                                 # final LN = last block
            if rest.startswith('blocks.'):
                block_idx = int(rest.split('.')[1])
                return n_blocks - block_idx              # blocks[n-1]→1, blocks[0]→n
            return n_blocks + 1                          # patch_embed, cls_token, pos_embed

        depth_label = f'{n_blocks} blocks'

    def apply_wd(name: str, param: torch.Tensor) -> bool:
        if param.ndim < 2:
            return False
        if any(kw in name for kw in ('cls_token', 'pos_embed')):
            return False
        return True

    buckets: Dict[Tuple[int, bool], List] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lid     = get_layer_id(name)
        has_wd  = apply_wd(name, param)
        buckets.setdefault((lid, has_wd), []).append(param)

    param_groups = [
        {
            'params':       params,
            'lr':           base_lr * (decay ** lid),
            'weight_decay': wd if has_wd else 0.0,
        }
        for (lid, has_wd), params in sorted(buckets.items())
    ]

    lrs = sorted({g['lr'] for g in param_groups})
    print(f'LLRD: {len(param_groups)} param groups  '
          f'LR range [{lrs[0]:.2e} – {lrs[-1]:.2e}]  '
          f'(decay={decay}, {depth_label})')
    return param_groups
