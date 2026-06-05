#!/usr/bin/env python3
"""
Export a BirdCLEF checkpoint to ONNX.

Usage:
    # Default: export for 5s inference chunks (CFG.chunk_duration)
    python export_onnx.py --exp_dir experiments/exp43

    # Export for 20s sliding-window inference (matches training duration)
    python export_onnx.py --exp_dir experiments/exp43 --duration 20
    # → writes swa_model_20s.onnx alongside the default swa_model.onnx

    python export_onnx.py --exp_dir experiments/exp43 --checkpoint best_model.pth
    python export_onnx.py --exp_dir experiments/exp43 --output /tmp/exp43_swa.onnx

Outputs <checkpoint_stem>.onnx (5s) or <checkpoint_stem>_<duration>s.onnx (other)
in the experiment directory by default.

The ONNX model:
  - Input  : mel_spec       (N, 3, n_mels, input_frames)  — batch dim is dynamic
  - Output : att_clipwise   (N, num_classes)               — raw logits, sigmoid NOT included
             so temperature scaling can be applied externally in the inference script

Requires: onnx, onnxruntime  (pip install onnx onnxruntime)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── codebase on path ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent / 'Codebase'
sys.path.insert(0, str(_ROOT))
from birdclef.config import Config
from birdclef.model import BirdCLEFModel


# ── helpers ───────────────────────────────────────────────────────────────────

def _config_from_json(json_path: Path) -> Config:
    """
    Reconstruct a Config from a saved run_config.json.
    Skips derived properties (img_size, img_size_chunk, is_vit) and
    any unknown keys added by future versions.
    """
    with open(json_path) as f:
        d = json.load(f)
    cfg  = Config()
    skip = {'img_size', 'img_size_chunk', 'is_vit'}
    for k, v in d.items():
        if k in skip:
            continue
        try:
            setattr(cfg, k, v)
        except AttributeError:
            pass   # read-only property; ignore
    return cfg


class _AttClipper(nn.Module):
    """
    Wrapper for standard 5s inference: returns att_clipwise (N, C).
    Used when export_duration == chunk_duration.
    """

    def __init__(self, model: BirdCLEFModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, att_clipwise, _ = self.model(x)
        return att_clipwise


class _FrameLogitsClipper(nn.Module):
    """
    Wrapper for sliding-window inference (export_duration > chunk_duration).

    Returns raw frame_logits (N, T_train, C) so the inference script can apply
    the 1st-place overlap-average-max reconstruction across all 12 windows.
    The reconstruction averages each temporal position over the multiple windows
    that cover it, then max-pools within each 5s chunk — eliminating the
    contamination that hurt the naive first-T_chunk approach.

    Output shape (N, T_train, C) — inference script detects the 3-D output and
    routes through _overlap_average_max instead of the 5s sigmoid path.
    """

    def __init__(self, model: BirdCLEFModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, frame_logits = self.model(x)   # (N, T_train, C)
        return frame_logits


# ── ViT ONNX compatibility fix ────────────────────────────────────────────────

@contextlib.contextmanager
def _no_antialias_bicubic():
    """
    Patch timm's ViT pos-embed resampling to use antialias=False during export.

    Root cause: timm >= 1.0 unconditionally calls resample_abs_pos_embed with
    antialias=True inside VisionTransformer._pos_embed whenever
    dynamic_img_size=True, even when the input size exactly matches the
    training size.  This emits aten::_upsample_bicubic2d_aa, which has no
    ONNX opset-17 implementation.

    Fix: monkeypatch the name 'resample_abs_pos_embed' in the
    timm.models.vision_transformer module dict — the one place the method
    looks it up at call time — to forward to the real function with
    antialias=False.  aten::_upsample_bicubic2d (no _aa suffix) IS in opset 17.

    Quality impact: none.  The pos-embed is always resampled to the same grid
    size it was trained at, so the interpolation is a mathematical identity
    regardless of whether anti-aliasing is applied.
    """
    try:
        import timm.models.vision_transformer as _vit_mod
        _orig = _vit_mod.resample_abs_pos_embed

        def _patched(*args, **kwargs):
            kwargs['antialias'] = False
            return _orig(*args, **kwargs)

        _vit_mod.resample_abs_pos_embed = _patched
        yield
    except AttributeError:
        # Older timm that doesn't have resample_abs_pos_embed in this module
        yield
    finally:
        try:
            _vit_mod.resample_abs_pos_embed = _orig  # type: ignore[possibly-undefined]
        except NameError:
            pass


# ── main export function ──────────────────────────────────────────────────────

def export_onnx(
    exp_dir: Path,
    checkpoint_name: str = 'swa_model.pth',
    output_path: Path | None = None,
    opset: int = 17,
    verify: bool = True,
    duration: int | None = None,
) -> Path:
    exp_dir = Path(exp_dir)
    cfg     = _config_from_json(exp_dir / 'run_config.json')

    label_map   = np.load(exp_dir / 'label_map.npy', allow_pickle=True).item()
    num_classes = len(label_map)

    print(f'Experiment   : {exp_dir.name}')
    print(f'Backbone     : {cfg.backbone}')
    print(f'Num classes  : {num_classes}')
    print(f'GeM pooling  : {cfg.use_gem}')
    print(f'ViT backbone : {cfg.is_vit}')

    model = BirdCLEFModel(cfg, num_classes, pretrained=False)
    ckpt_path = exp_dir / checkpoint_name
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    print(f'Checkpoint   : {ckpt_path.name}  → loaded')

    # Input frame count: use the requested duration, defaulting to chunk_duration (5s).
    # Pass --duration 20 (or cfg.duration) to export a model for 20s sliding-window
    # inference as used by birdclef_inference_0417.py with TRAIN_DURATION=20.
    export_duration = duration if duration is not None else cfg.chunk_duration
    natural_frames  = math.floor(cfg.sr * export_duration / cfg.hop_length) + 1
    # When mel_w is set, preprocessing resizes the time axis to mel_w before the
    # backbone.  For the standard chunk-duration export, use mel_w as the dummy
    # width so the ONNX input shape matches inference.  For non-standard durations
    # (sliding-window), fall back to natural frames — mel_w was only applied to
    # chunk-duration clips during training.
    input_frames    = (cfg.mel_w
                       if cfg.mel_w is not None and export_duration == cfg.chunk_duration
                       else natural_frames)
    dummy           = torch.zeros(1, 3, cfg.n_mels, input_frames)
    if cfg.mel_w is not None and export_duration == cfg.chunk_duration:
        print(f'mel_w resize : natural {natural_frames} → {cfg.mel_w} frames (hop={cfg.hop_length})')
    print(f'Export dur.  : {export_duration}s  ({input_frames} mel frames)')
    print(f'Input shape  : (N, 3, {cfg.n_mels}, {input_frames})  [N is dynamic]')

    if export_duration > cfg.chunk_duration:
        with torch.no_grad():
            _, _, fl = model(dummy)
        T_train = fl.shape[1]
        T_chunk = max(1, T_train * cfg.chunk_duration // export_duration)
        print(f'T_train      : {T_train}  (backbone frames for {export_duration}s window)')
        print(f'T_chunk      : {T_chunk}  (frames per {cfg.chunk_duration}s chunk)')
        print(f'Output       : frame_logits (N, {T_train}, C) — overlap reconstruction in inference script')
        wrapper = _FrameLogitsClipper(model)
    else:
        wrapper = _AttClipper(model)

    # IMPORTANT: must call wrapper.eval() explicitly.
    # torch.onnx.export re-applies the training state from the top-level
    # module, and the wrapper defaults to training=True (standard nn.Module
    # behaviour), even though the inner model was already set to eval.
    # Without this call, BatchNorm layers export with train=True and compute
    # batch statistics at inference instead of running statistics — large errors.
    wrapper.eval()

    # Confirm every BN layer is now in eval mode
    bn_in_train = [
        n for n, m in wrapper.named_modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        and m.training
    ]
    if bn_in_train:
        raise RuntimeError(
            'BN layers still in training mode after wrapper.eval(); '
            'they would export with train=True and produce wrong outputs.\n'
            'Affected: ' + ', '.join(bn_in_train)
        )

    if output_path is None:
        stem = Path(checkpoint_name).stem
        # Append duration suffix when it differs from the default 5s chunk to
        # keep the standard swa_model.onnx name for 5s and avoid overwriting it.
        suffix = f'_{export_duration}s' if export_duration != cfg.chunk_duration else ''
        output_path = exp_dir / f'{stem}{suffix}.onnx'

    print(f'Exporting to : {output_path} ...')
    if cfg.is_vit:
        print('ViT backbone  : patching timm pos-embed resampling (antialias=False) '
              'for ONNX opset compatibility')

    # training=TrainingMode.EVAL tells the tracer to force eval-mode semantics for
    # every operator, including BatchNormalization.  Without this, the ONNX symbolic
    # function for BN can emit training_mode=1 nodes (5 outputs) depending on Python's
    # warning filter state — OpenVINO's ONNX frontend rejects these unconditionally.
    # NOTE: do NOT wrap torch.onnx.export in warnings.catch_warnings(); doing so
    # creates an isolated warning-filter scope that disrupts PyTorch's internal BN
    # training-mode detection and reproducibly causes training_mode=1 export for CNN
    # backbones even when the model is in eval mode.
    out_name = 'frame_logits' if export_duration > cfg.chunk_duration else 'att_clipwise'
    _vit_ctx = _no_antialias_bicubic() if cfg.is_vit else contextlib.nullcontext()
    with _vit_ctx:
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            input_names=['mel_spec'],
            output_names=[out_name],
            dynamic_axes={
                'mel_spec': {0: 'batch_size'},
                out_name:   {0: 'batch_size'},
            },
            opset_version=opset,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )

    size_mb = output_path.stat().st_size / 1e6
    print(f'Saved        : {output_path}  ({size_mb:.1f} MB)')

    if verify:
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            print('WARNING: onnx / onnxruntime not installed — skipping verification.')
            print('         pip install onnx onnxruntime')
            return output_path

        onnx.checker.check_model(onnx.load(str(output_path)))
        print('ONNX graph   : OK')

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(output_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider'],
        )

        # Compare on a realistic batch of 12 chunks.
        # IMPORTANT: use uniform [0,1] single-channel-repeated input, not torch.randn.
        # The model's BatchNorm running stats were computed on [0,1] mel spectrograms,
        # so out-of-distribution random normal inputs produce huge intermediate
        # activations (~1e4–1e5 logits).  At that scale, even correct floating-point
        # arithmetic accumulates absolute errors that exceed the 1e-4 threshold — a
        # false alarm.  In-distribution [0,1] inputs keep logits in the ~[-50, 50]
        # range where a 1e-4 absolute threshold is meaningful.
        # All channels are identical (single mel repeated × in_chans) to match the
        # actual data format the model was trained on.
        single_ch  = torch.rand(12, 1, cfg.n_mels, input_frames)
        test_input = single_ch.repeat(1, cfg.in_chans, 1, 1)
        with torch.inference_mode():
            pt_out  = wrapper(test_input).numpy()
        ort_out = session.run(None, {'mel_spec': test_input.numpy()})[0]

        max_diff  = float(np.abs(pt_out - ort_out).max())
        mean_diff = float(np.abs(pt_out - ort_out).mean())
        print(f'PyTorch vs ONNX :  max|Δ| = {max_diff:.2e}   mean|Δ| = {mean_diff:.2e}')
        if max_diff > 1e-4:
            print(f'WARNING: numerical mismatch larger than expected ({max_diff:.2e})')
        else:
            print('Numerical check  : PASSED')

    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Export BirdCLEF checkpoint to ONNX')
    parser.add_argument(
        '--exp_dir', required=True,
        help='Experiment directory (must contain run_config.json, label_map.npy, <checkpoint>)')
    parser.add_argument(
        '--checkpoint', default='swa_model.pth',
        help='Checkpoint filename within exp_dir (default: swa_model.pth)')
    parser.add_argument(
        '--output', default=None,
        help='Output .onnx path (default: exp_dir/<checkpoint_stem>.onnx)')
    parser.add_argument(
        '--opset', type=int, default=17,
        help='ONNX opset version (default: 17)')
    parser.add_argument(
        '--no_verify', action='store_true',
        help='Skip numerical verification against PyTorch')
    parser.add_argument(
        '--duration', type=int, default=None,
        help='Input window duration in seconds (default: chunk_duration from run_config, '
             'typically 5). Pass 20 to export for sliding 20s-window inference — '
             'output will be named <checkpoint_stem>_20s.onnx to avoid overwriting '
             'the standard 5s export.')
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    export_onnx(
        exp_dir         = args.exp_dir,
        checkpoint_name = args.checkpoint,
        output_path     = output_path,
        opset           = args.opset,
        verify          = not args.no_verify,
        duration        = args.duration,
    )


if __name__ == '__main__':
    main()
