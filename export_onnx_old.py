#!/usr/bin/env python3
"""
Export a BirdCLEF checkpoint to ONNX.

Usage:
    python export_onnx.py --exp_dir experiments/exp25_supervised
    python export_onnx.py --exp_dir experiments/exp25_supervised --checkpoint best_model.pth
    python export_onnx.py --exp_dir experiments/exp25_supervised --output /tmp/exp25_swa.onnx

Outputs <checkpoint_stem>.onnx in the experiment directory by default.

The ONNX model:
  - Input  : mel_spec       (N, 3, n_mels, chunk_frames)  — batch dim is dynamic
  - Output : att_clipwise   (N, num_classes)               — raw logits, sigmoid NOT included
             so temperature scaling can be applied externally in the inference script

Requires: onnx, onnxruntime  (pip install onnx onnxruntime)
"""

from __future__ import annotations

import argparse
import json
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
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except AttributeError:
                pass   # read-only property; ignore
    return cfg


class _AttClipper(nn.Module):
    """
    Thin wrapper that returns only att_clipwise (output index 1) from
    BirdCLEFModel.  The two other outputs (clipwise_logits, frame_logits)
    are not needed at inference time and are dropped here so the ONNX graph
    is smaller and avoids exporting unused computation.
    """

    def __init__(self, model: BirdCLEFModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, att_clipwise, _ = self.model(x)
        return att_clipwise


# ── main export function ──────────────────────────────────────────────────────

def export_onnx(
    exp_dir: Path,
    checkpoint_name: str = 'swa_model.pth',
    output_path: Path | None = None,
    opset: int = 17,
    verify: bool = True,
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

    wrapper = _AttClipper(model)
    # IMPORTANT: must call wrapper.eval() explicitly.
    # torch.onnx.export re-applies the training state from the top-level
    # module, and _AttClipper.__init__ defaults to training=True (standard
    # nn.Module behaviour), even though the inner model was already set to
    # eval.  Without this call, BatchNorm layers are exported with train=True
    # and will compute batch statistics at inference instead of using the
    # trained running statistics — causing large numerical errors (|Δ|>>1).
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

    chunk_frames = cfg.img_size_chunk[1]
    dummy        = torch.zeros(1, 3, cfg.n_mels, chunk_frames)
    print(f'Input shape  : (N, 3, {cfg.n_mels}, {chunk_frames})  [N is dynamic]')

    if output_path is None:
        stem        = Path(checkpoint_name).stem
        output_path = exp_dir / f'{stem}.onnx'

    print(f'Exporting to : {output_path} ...')
    # Suppress the "batch_norm train=True" UserWarning: it is a false positive.
    # PyTorch's tracer records BN with train=True in the intermediate trace, which
    # triggers this diagnostic, but do_constant_folding then folds the running
    # mean/var into the graph as constants — the exported model is eval-mode correct,
    # as confirmed by the numerical verification below.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            'ignore',
            message='.*batch_norm.*train=True.*',
            category=UserWarning,
        )
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            input_names=['mel_spec'],
            output_names=['att_clipwise'],
            dynamic_axes={
                'mel_spec':     {0: 'batch_size'},
                'att_clipwise': {0: 'batch_size'},
            },
            opset_version=opset,
            do_constant_folding=True,
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

        # Compare on a realistic batch of 12 chunks
        test_input = torch.randn(12, 3, cfg.n_mels, chunk_frames)
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
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    export_onnx(
        exp_dir         = args.exp_dir,
        checkpoint_name = args.checkpoint,
        output_path     = output_path,
        opset           = args.opset,
        verify          = not args.no_verify,
    )


if __name__ == '__main__':
    main()
