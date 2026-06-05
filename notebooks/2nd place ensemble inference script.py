# %% [code] {"jupyter":{"outputs_hidden":false}}
n_dry_run=10 #Use more for timing tests

# %% [markdown]
# # PERCH SECTION
# Copied with minor adaptations from [this high-scoring public notebook](https://www.kaggle.com/code/mtoshidesu/test-0-948) and [this one](https://www.kaggle.com/code/youssefmo942009/lb-0-948?scriptVersionId=319828635). 0.936 LB 0.930 PB

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ── Perch-stack submission notebook ───────────────────────────────────────────
# Purpose : Load pre-trained artifacts, run Perch on 600 hidden test files,
#           apply the full post-processing stack, write submission CSV.
# Inputs  : perch-artefacts-birdclef2026 dataset (from perch_artifacts_train.py)
#           Perch ONNX or TF SavedModel dataset
#           Competition data (test_soundscapes/)
# Runtime : ~15-18 min CPU (dominated by Perch inference on 600 files).

import gc
import os
import random
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import concurrent.futures
import torch.nn as nn
import torch.nn.functional as F
import joblib

INPUT_ROOT = Path("/kaggle/input")


def find_wheel(pattern):
    for p in INPUT_ROOT.rglob(pattern):
        return p
    raise FileNotFoundError(pattern)


ONNX_WHL = Path(
    "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/"
    "onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
)
if ONNX_WHL.exists():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(ONNX_WHL)],
        check=True,
    )
    print("ONNX Runtime installed")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     str(find_wheel("tensorboard-2.20.0-*.whl"))], check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     str(find_wheel("tensorflow-2.20.0-*.whl"))], check=True,
)
print("TF 2.20 installed")

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
    print("ONNX Runtime available")
except ImportError:
    _ONNX_AVAILABLE = False
    print("ONNX not available, falling back to TF")

try:
    import openvino as ov
    _OV_AVAILABLE = True
except ImportError:
    _OV_AVAILABLE = False
    
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(4)
tf.random.set_seed(4)
torch.use_deterministic_algorithms(True)
print("Global random seed set to 4")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

tf.experimental.numpy.experimental_enable_numpy_behavior()
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

_WALL_START = time.time()

BASE = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path(
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
)
WORK_DIR = Path("/kaggle/working/cache")
WORK_DIR.mkdir(parents=True, exist_ok=True)

ARTEFACT_DIR = Path("/kaggle/input/datasets/tennogh/birdclef2026-2nd-place-models/perch_artefacts")  # adjust slug if needed

SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12

BATCH_FILES = 8  # memory-safe batch size for Perch inference

# Shared in-memory cache: skip writing/reading 7 intermediate CSVs (~40-60 sec saved).
# Each section stores (row_ids_array, probs_array) here; ensembling reads from it.
_PROBS_CACHE = {}  # {model_name: (row_ids_1d, probs_2d float32)}

print("Constants set")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ── Data ──────────────────────────────────────────────────────────────────────
taxonomy = pd.read_csv(BASE / "taxonomy.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def parse_fname(name):
    m = FNAME_RE.match(name)
    if not m:
        return {"site": "unknown", "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}


def union_labels(series):
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t:
                    out.add(t)
    return sorted(out)


sc = (
    soundscape_labels.groupby(["filename", "start", "end"])["primary_label"]
    .apply(union_labels)
    .reset_index(name="label_list")
)

sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc[
    "end_sec"
].astype(str)

_meta = sc["filename"].apply(parse_fname).apply(pd.Series)
sc = pd.concat([sc, _meta], axis=1)

Y_SC = np.zeros((len(sc), N_CLASSES), dtype=np.uint8)
for i, lbls in enumerate(sc["label_list"]):
    for lbl in lbls:
        if lbl in label_to_idx:
            Y_SC[i, label_to_idx[lbl]] = 1

print(f"Classes: {N_CLASSES}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ── Perch backbone ────────────────────────────────────────────────────────────
ONNX_PERCH_PATH = next(
    INPUT_ROOT.glob("**/perch_v2_no_dft*.onnx"),
    next(INPUT_ROOT.glob("**/perch_v2*.onnx"), Path("")),
)
USE_ONNX = _ONNX_AVAILABLE and ONNX_PERCH_PATH.exists()

if USE_ONNX:
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = 4
    _so.inter_op_num_threads = 4
    _so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    _so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ONNX_SESSION = ort.InferenceSession(
        str(ONNX_PERCH_PATH),
        sess_options=_so,
        providers=["CPUExecutionProvider"],
    )
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_OUT_MAP = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
    print(f"Using ONNX Perch: {ONNX_PERCH_PATH.name}")
else:
    print("Using TF SavedModel Perch")
    birdclassifier = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = birdclassifier.signatures["serving_default"]

bc_labels = (
    pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
    .reset_index()
    .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
)
NO_LABEL = len(bc_labels)

mapping = taxonomy.merge(
    bc_labels.rename(columns={"scientific_name": "scientific_name"}),
    on="scientific_name",
    how="left",
)
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
lbl2bc = mapping.set_index("primary_label")["bc_index"]

BC_INDICES = np.array([int(lbl2bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK = BC_INDICES != NO_LABEL
MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)

print(f"Mapped: {MAPPED_MASK.sum()} / {N_CLASSES} species have a Perch logit")

import re as _re

UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA = {"Amphibia", "Insecta"}

proxy_map = {}
unmapped_df = taxonomy[
    taxonomy["primary_label"].isin([PRIMARY_LABELS[i] for i in UNMAPPED_POS])
].copy()

for _, row in unmapped_df.iterrows():
    target = row["primary_label"]
    sci = str(row["scientific_name"])
    genus = sci.split()[0]
    hits = bc_labels[
        bc_labels["scientific_name"].astype(str).str.match(rf"^{_re.escape(genus)}\s", na=False)
    ]
    if len(hits) > 0:
        proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()

PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
proxy_map = {
    idx: bc_idxs
    for idx, bc_idxs in proxy_map.items()
    if CLASS_NAME_MAP.get(PRIMARY_LABELS[idx]) in PROXY_TAXA
}

print(
    f"Unmapped: {len(UNMAPPED_POS)} | Proxy: {len(proxy_map)} | No signal: {len(UNMAPPED_POS) - len(proxy_map)}"
)

temperatures = np.ones(N_CLASSES, dtype=np.float32)
for ci, label in enumerate(PRIMARY_LABELS):
    cls = CLASS_NAME_MAP.get(label, "Aves")
    temperatures[ci] = 0.95 if cls in TEXTURE_TAXA else 1.10


def read_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:
        y = y[:FILE_SAMPLES]
    return y


def run_perch(paths, batch_files=16, verbose=True):
    paths = [Path(p) for p in paths]
    n_rows = len(paths) * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.zeros(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embs = np.zeros((n_rows, 1536), dtype=np.float32)

    wr = 0
    itr = tqdm(range(0, len(paths), batch_files), desc="Perch") if verbose else range(
        0, len(paths), batch_files
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:
        next_paths = paths[0:batch_files]
        future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

        for start in itr:
            batch_paths = next_paths
            batch_n = len(batch_paths)
            batch_audio = [f.result() for f in future_audio]

            next_start = start + batch_files
            if next_start < len(paths):
                next_paths = paths[next_start: next_start + batch_files]
                future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

            x = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            br = wr

            for bi, path in enumerate(batch_paths):
                y = batch_audio[bi]
                meta = parse_fname(path.name)
                stem = path.stem

                x[bi * N_WINDOWS: (bi + 1) * N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
                row_ids[wr: wr + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[wr: wr + N_WINDOWS] = path.name
                sites[wr: wr + N_WINDOWS] = meta["site"]
                hours[wr: wr + N_WINDOWS] = meta["hour_utc"]
                wr += N_WINDOWS

            if USE_ONNX:
                outs = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
                logits = outs[ONNX_OUT_MAP["label"]].astype(np.float32)
                emb = outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)
            else:
                out = infer_fn(inputs=tf.convert_to_tensor(x))
                logits = out["label"].numpy().astype(np.float32)
                emb = out["embedding"].numpy().astype(np.float32)

            scores[br:wr, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
            embs[br:wr] = emb

            for pos_idx, bc_idxs in proxy_map.items():
                bc_arr = np.array(bc_idxs, dtype=np.int32)
                scores[br:wr, pos_idx] = logits[:, bc_arr].max(axis=1)

            del x, logits, emb, batch_audio
            gc.collect()

    meta_df = pd.DataFrame(
        {"row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours}
    )
    return meta_df, scores, embs


print("Perch inference engine defined")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ── Post-processing helpers ───────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def build_prior_tables(sc_df, Y_labels):
    sc_df = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)

    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]
        mask = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_labels[mask].mean(axis=0)

    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]
        mask = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_labels[mask].mean(axis=0)

    sh_keys = sorted(
        {
            (str(s), int(h))
            for s, h in zip(sc_df["site"].dropna(), sc_df["hour_utc"].dropna())
            if not pd.isna(s) and not pd.isna(h)
        }
    )
    sh_to_i = {k: i for i, k in enumerate(sh_keys)}
    sh_p = np.zeros((len(sh_keys), Y_labels.shape[1]), dtype=np.float32)
    sh_n = np.zeros(len(sh_keys), dtype=np.float32)
    for (s, h) in sh_keys:
        i = sh_to_i[(s, h)]
        mask = (sc_df["site"].astype(str).values == s) & (
            sc_df["hour_utc"].astype(int).values == h
        )
        sh_n[i] = mask.sum()
        sh_p[i] = Y_labels[mask].mean(axis=0)

    if len(hour_keys) >= 3:
        _full_hour_p = np.zeros((24, hour_p.shape[1]), dtype=np.float32)
        for _h, _i in hour_to_i.items():
            _full_hour_p[int(_h)] = hour_p[_i]
        _tiled = np.tile(_full_hour_p, (3, 1))
        _tiled_smooth = gaussian_filter1d(_tiled, sigma=1.5, axis=0, mode='wrap')
        _full_smooth = _tiled_smooth[24:48]
        for _h, _i in hour_to_i.items():
            hour_p[_i] = _full_smooth[int(_h)]
        hour_p = np.clip(hour_p, 0.0, 1.0)

    return {
        "global_p": global_p,
        "site_to_i": site_to_i, "site_p": site_p, "site_n": site_n,
        "hour_to_i": hour_to_i, "hour_p": hour_p, "hour_n": hour_n,
        "sh_to_i": sh_to_i, "sh_p": sh_p, "sh_n": sh_n,
    }


def apply_prior(scores, sites, hours, tables, lambda_prior=0.4):
    eps = 1e-4
    n = len(scores)
    out = scores.copy()
    p = np.tile(tables["global_p"], (n, 1))

    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j = tables["hour_to_i"][h]
            nh = tables["hour_n"][j]
            w = nh / (nh + 8.0)
            p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]

    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j = tables["site_to_i"][s]
            ns = tables["site_n"][j]
            w = ns / (ns + 8.0)
            p[i] = w * tables["site_p"][j] + (1 - w) * p[i]

    if "sh_to_i" in tables:
        for i, (s, h) in enumerate(zip(sites, hours)):
            key = (str(s), int(h))
            if key in tables["sh_to_i"]:
                j = tables["sh_to_i"][key]
                nsh = tables["sh_n"][j]
                w = nsh / (nsh + 4.0)
                p[i] = w * tables["sh_p"][j] + (1 - w) * p[i]

    p = np.clip(p, eps, 1 - eps)
    out += lambda_prior * (np.log(p) - np.log1p(-p))
    return out.astype(np.float32)


def file_confidence_scale(probs, n_windows=12, top_k=2, power=0.4):
    N, C = probs.shape
    view = probs.reshape(-1, n_windows, C)
    sorted_v = np.sort(view, axis=1)
    top_k_mean = sorted_v[:, -top_k:, :].mean(axis=1, keepdims=True)
    return (view * np.power(top_k_mean, power)).reshape(N, C)


def rank_aware_scaling(probs, n_windows=12, power=0.4):
    N, C = probs.shape
    view = probs.reshape(-1, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)
    return (view * np.power(file_max, power)).reshape(N, C)


def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    N, C = probs.shape
    result = probs.copy()
    view = probs.reshape(-1, n_windows, C)
    out = result.reshape(-1, n_windows, C)
    for t in range(n_windows):
        conf = view[:, t, :].max(axis=-1, keepdims=True)
        alpha = base_alpha * (1.0 - conf)
        if t == 0:
            neighbor_avg = (view[:, t, :] + view[:, t + 1, :]) / 2.0
        elif t == n_windows - 1:
            neighbor_avg = (view[:, t - 1, :] + view[:, t, :]) / 2.0
        else:
            neighbor_avg = (view[:, t - 1, :] + view[:, t + 1, :]) / 2.0
        out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * neighbor_avg
    return result


def apply_per_class_thresholds(scores, thresholds):
    t = thresholds[None, :]   # (1, C) broadcast
    above = scores > t
    scaled = np.where(above,
                      0.5 + 0.5 * (scores - t) / (1 - t + 1e-8),
                      0.5 * scores / (t + 1e-8))
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ── SSM architecture (definitions only — weights loaded from artifact) ─────────
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_sz, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = F.silu(self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2))
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        h = torch.zeros(B_sz, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))
        return torch.stack(ys, dim=1) + x * self.D[None, None, :]


class LightProtoSSM(nn.Module):
    def __init__(
        self,
        d_input=1536,
        d_model=128,
        d_state=16,
        n_classes=234,
        n_windows=12,
        dropout=0.15,
        n_sites=20,
        meta_dim=16,
        use_cross_attn=True,
        cross_attn_heads=2,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.ssm_fwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_bwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(2)])
        self.ssm_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.drop = nn.Dropout(dropout)

        if use_cross_attn:
            self.cross_attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(d_model, cross_attn_heads, dropout=dropout, batch_first=True)
                    for _ in range(2)
                ]
            )
            self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            meta = self.meta_proj(
                torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1)
            )
            h = h + meta[:, None, :]

        for i, (fwd, bwd, merge, norm) in enumerate(
            zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm)
        ):
            res = h
            hf = fwd(h)
            hb = bwd(h.flip(1)).flip(1)
            h = self.drop(merge(torch.cat([hf, hb], dim=-1)))
            h = norm(h + res)

            if self.use_cross_attn:
                attn_out, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + attn_out)

        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(h_n, p_n.T) * F.softplus(self.proto_temp) + self.class_bias[None, None, :]

        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            out = alpha * sim + (1 - alpha) * perch_logits
        else:
            out = sim
        return out


class ResidualSSM(nn.Module):
    def __init__(
        self,
        d_input=1536,
        d_scores=234,
        d_model=64,
        d_state=8,
        n_classes=234,
        n_windows=12,
        dropout=0.1,
        n_sites=20,
        meta_dim=8,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm_drop = nn.Dropout(dropout)
        self.output_head = nn.Linear(d_model, n_classes)

        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, emb, first_pass, site_ids=None, hours=None):
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            meta = self.meta_proj(
                torch.cat(
                    [
                        self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1)),
                        self.hour_emb(hours.clamp(0, 23)),
                    ],
                    dim=-1,
                )
            )
            h = h + meta.unsqueeze(1)

        res = h
        hf = self.ssm_fwd(h)
        hb = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.ssm_drop(self.ssm_merge(torch.cat([hf, hb], dim=-1)))
        h = self.ssm_norm(h + res)
        return self.output_head(h)


class VectorizedMLPProbes(nn.Module):
    def __init__(self, probe_models):
        super().__init__()
        self.valid_classes = sorted(probe_models.keys())
        V = len(self.valid_classes)

        if V == 0:
            self.weights = nn.ParameterList()
            self.biases = nn.ParameterList()
            self.n_layers = 0
            return

        sample = probe_models[self.valid_classes[0]]
        self.n_layers = len(sample.coefs_)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for li in range(self.n_layers):
            W = np.stack([probe_models[c].coefs_[li] for c in self.valid_classes], axis=0)
            b = np.stack([probe_models[c].intercepts_[li] for c in self.valid_classes], axis=0)
            self.weights.append(
                nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False)
            )
            self.biases.append(
                nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=False)
            )

    def forward(self, x):
        h = x
        for i in range(self.n_layers):
            h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
            if i < self.n_layers - 1:
                h = torch.relu(h)
        return h.squeeze(-1)


def _run_probe_group(group_models, valid_classes_group, scores_test, Z_test, N):
    Vg = len(valid_classes_group)
    raw_g = scores_test[:, valid_classes_group].T
    n_files = N // N_WINDOWS
    raw_view_g = raw_g.reshape(Vg, n_files, N_WINDOWS)

    prev_g = np.concatenate([raw_view_g[:, :, :1], raw_view_g[:, :, :-1]], axis=2).reshape(Vg, N)
    nxt_g = np.concatenate([raw_view_g[:, :, 1:], raw_view_g[:, :, -1:]], axis=2).reshape(Vg, N)
    mean_g = np.repeat(raw_view_g.mean(axis=2), N_WINDOWS, axis=1)
    mx_g = np.repeat(raw_view_g.max(axis=2), N_WINDOWS, axis=1)
    std_g = np.repeat(raw_view_g.std(axis=2), N_WINDOWS, axis=1)

    scalar_g = np.stack([raw_g, prev_g, nxt_g, mean_g, mx_g, std_g], axis=-1).astype(np.float32)
    Z_exp_g = np.broadcast_to(Z_test, (Vg, N, Z_test.shape[1]))
    X_g = np.concatenate([Z_exp_g.astype(np.float32), scalar_g], axis=-1)

    vec_probe = VectorizedMLPProbes(group_models).eval()
    with torch.no_grad():
        preds_g = vec_probe(torch.tensor(X_g)).numpy()
    return preds_g


def apply_mlp_probes_vectorized(emb_test, scores_test, probe_models, scaler, pca, alpha_blend=0.4):
    if len(probe_models) == 0:
        return scores_test.copy()

    Z_test = pca.transform(scaler.transform(emb_test)).astype(np.float32)
    N = len(scores_test)
    result = scores_test.copy()

    def _arch_key(clf):
        return tuple(w.shape[1] for w in clf.coefs_)

    from collections import defaultdict
    groups = defaultdict(dict)
    for ci, clf in probe_models.items():
        groups[_arch_key(clf)][ci] = clf

    for arch, group_models in groups.items():
        valid_classes_group = sorted(group_models.keys())
        preds_g = _run_probe_group(group_models, valid_classes_group, scores_test, Z_test, N)
        result[:, valid_classes_group] = (
            (1.0 - alpha_blend) * scores_test[:, valid_classes_group]
            + alpha_blend * preds_g.T
        )

    return result


def run_tta_proto(proto_model, emb_files, sc_files, site_t, hour_t, shifts=[0, 1, -1, 2, -2]):
    proto_model.eval()
    all_preds = []

    emb_t = torch.tensor(emb_files, dtype=torch.float32)
    sc_t = torch.tensor(sc_files, dtype=torch.float32)

    for shift in shifts:
        e = torch.roll(emb_t, shift, dims=1) if shift else emb_t
        s = torch.roll(sc_t, shift, dims=1) if shift else sc_t
        with torch.no_grad():
            out = proto_model(e, s, site_ids=site_t, hours=hour_t).numpy()
        if shift:
            out = np.roll(out, -shift, axis=1)
        all_preds.append(out)

    with torch.no_grad():
        out_flip = proto_model(
            emb_t.flip(1), sc_t.flip(1), site_ids=site_t, hours=hour_t
        ).numpy()
    all_preds.append(out_flip[:, ::-1, :].copy())

    return np.mean(all_preds, axis=0)


print("All inference functions defined")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ── Load pre-trained artifacts ─────────────────────────────────────────────────
_saved_labels = np.load(ARTEFACT_DIR / "primary_labels.npy", allow_pickle=True).tolist()
if _saved_labels != PRIMARY_LABELS:
    raise RuntimeError(
        f"Artifact label schema ({len(_saved_labels)} classes) does not match "
        f"this notebook ({len(PRIMARY_LABELS)}). Re-run perch_artifacts_train.py."
    )
print("Label schema verified")

_proto_ckpt = torch.load(ARTEFACT_DIR / "proto_ssm.pt", map_location="cpu")
proto_model = LightProtoSSM(
    n_classes=N_CLASSES,
    n_sites=_proto_ckpt["n_sites"],
    use_cross_attn=True,
    cross_attn_heads=2,
)
proto_model.load_state_dict(_proto_ckpt["state_dict"])
proto_model.eval()
site2i_tr = _proto_ckpt["site2i_tr"]
n_sites_cap = _proto_ckpt["n_sites"]
print("ProtoSSM loaded")

_res_ckpt = torch.load(ARTEFACT_DIR / "residual_ssm.pt", map_location="cpu")
res_model = ResidualSSM(n_classes=N_CLASSES)
res_model.load_state_dict(_res_ckpt["state_dict"])
res_model.eval()
correction_weight = _res_ckpt["correction_weight"]
print(f"ResidualSSM loaded  (correction_weight={correction_weight:.2f})")

_probes = joblib.load(ARTEFACT_DIR / "mlp_probes.pkl")
probe_models = _probes["probe_models"]
emb_scaler = _probes["emb_scaler"]
emb_pca = _probes["emb_pca"]
alpha_blend = _probes["alpha_blend"]
print(f"MLP probes loaded   ({len(probe_models)} probes, alpha_blend={alpha_blend})")

PER_CLASS_THRESHOLDS = np.load(ARTEFACT_DIR / "per_class_thresholds.npy")
print(f"Thresholds loaded   (mean={PER_CLASS_THRESHOLDS.mean():.3f})")

print(f"Artifacts loaded in {(time.time() - _WALL_START):.1f}s")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ── Test inference ─────────────────────────────────────────────────────────────
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
IS_DRY_RUN = len(test_paths) == 0
if IS_DRY_RUN:
    n = n_dry_run
    print(f"No hidden test — dry-run on {n} train files")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:n]
else:
    print(f"Hidden test files: {len(test_paths)}")

t_perch = time.time()
meta_te, sc_te, emb_te = run_perch(test_paths, BATCH_FILES, verbose=True)
print(f"Perch inference: {(time.time() - t_perch) / 60:.1f} min  scores: {sc_te.shape}")

# ── Free Perch backbone immediately after inference ───────────────────────────
if USE_ONNX:
    del ONNX_SESSION
else:
    del birdclassifier, infer_fn
gc.collect()
print("Perch backbone freed")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ── Apply full post-processing stack ──────────────────────────────────────────
n_test_files = len(sc_te) // N_WINDOWS
emb_te_f = emb_te.reshape(n_test_files, N_WINDOWS, -1)
sc_te_f = sc_te.reshape(n_test_files, N_WINDOWS, -1)

test_fnames = meta_te.drop_duplicates("filename")["filename"].tolist()
test_site_ids = np.array(
    [
        min(site2i_tr.get(meta_te.loc[meta_te["filename"] == fn, "site"].iloc[0], 0), n_sites_cap - 1)
        for fn in test_fnames
    ],
    dtype=np.int64,
)
test_hour_ids = np.array(
    [int(meta_te.loc[meta_te["filename"] == fn, "hour_utc"].iloc[0]) % 24 for fn in test_fnames],
    dtype=np.int64,
)

proto_out = run_tta_proto(
    proto_model, emb_te_f, sc_te_f,
    site_t=torch.tensor(test_site_ids, dtype=torch.long),
    hour_t=torch.tensor(test_hour_ids, dtype=torch.long),
    shifts=[0, 1, -1, 2, -2],
)
proto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)

prior_tables = build_prior_tables(sc, Y_SC)
sc_te_adjusted = apply_prior(
    sc_te,
    sites=meta_te["site"].to_numpy(),
    hours=meta_te["hour_utc"].to_numpy(),
    tables=prior_tables,
    lambda_prior=0.4,
)
sc_te_adjusted = apply_mlp_probes_vectorized(
    emb_te, sc_te_adjusted, probe_models, emb_scaler, emb_pca, alpha_blend
)

ENSEMBLE_W_PER_CLASS = np.where(MAPPED_MASK, 0.60, 0.35).astype(np.float32)
first_pass_flat = (
    ENSEMBLE_W_PER_CLASS[None, :] * proto_scores_flat
    + (1.0 - ENSEMBLE_W_PER_CLASS)[None, :] * sc_te_adjusted
)
print(
    f"[Ensemble] mapped={ENSEMBLE_W_PER_CLASS[MAPPED_MASK].mean():.2f}  "
    f"unmapped={ENSEMBLE_W_PER_CLASS[~MAPPED_MASK].mean():.2f}"
)

first_pass_te_f = first_pass_flat.reshape(n_test_files, N_WINDOWS, -1)
res_model.eval()
with torch.no_grad():
    test_correction = res_model(
        torch.tensor(emb_te_f, dtype=torch.float32),
        torch.tensor(first_pass_te_f, dtype=torch.float32),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long),
        hours=torch.tensor(test_hour_ids, dtype=torch.long),
    ).numpy()

correction_flat = test_correction.reshape(-1, N_CLASSES).astype(np.float32)
final_scores = first_pass_flat + correction_weight * correction_flat
final_scores = final_scores / temperatures[None, :]

probs = sigmoid(final_scores)
probs = file_confidence_scale(probs, n_windows=N_WINDOWS, top_k=2, power=0.4)
probs = rank_aware_scaling(probs, n_windows=N_WINDOWS, power=0.4)
probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS, base_alpha=0.20)
probs = np.clip(probs, 0.0, 1.0)
probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)

sub = pd.DataFrame(probs.astype(np.float32), columns=PRIMARY_LABELS)
sub.insert(0, "row_id", meta_te["row_id"].values)
_PROBS_CACHE["perch"] = (meta_te["row_id"].values.copy(), probs.astype(np.float32))
print(f"Perch probs cached  shape={sub.shape}")
print(f"Total wall time: {(time.time() - _WALL_START) / 60:.1f} min")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ── Cleanup — free everything before SED ensemble cells ───────────────────────
del proto_model, res_model
del emb_te, sc_te, sc_te_adjusted, emb_te_f, sc_te_f
del probe_models, emb_scaler, emb_pca
del proto_out, proto_scores_flat, first_pass_flat, first_pass_te_f
del test_correction, correction_flat, final_scores, probs, sub
gc.collect()
print("Perch stack fully cleaned up. Ready for SED ensemble cell.")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # SED SECTION: 
# Inspired by the ["Distilled SED" public notebook](https://www.kaggle.com/code/tuckerarrants/bc2026-distilled-sed), but instead trained without KD on XC-pretrained v2_s. 0.929 LB 0.938 PB

# %% [code] {"jupyter":{"outputs_hidden":false}}
import librosa
from scipy.ndimage import gaussian_filter1d

N_MELS_SED = 128
N_FFT_SED  = 2048
HOP_SED    = 512
FMIN_SED   = 20
FMAX_SED   = 16000
TOP_DB_SED = 80


def find_sed_dir():
    hits = sorted(Path("/kaggle/input/datasets/tennogh/birdclef2026-2nd-place-models/v2s softAUC nodistill 5fold").rglob("sed_fold0.onnx"))
    if not hits:
        raise FileNotFoundError(
            "sed_fold0.onnx not found. "
            "Attach tuckerarrants/bc2026-distilled-sed-public to this notebook."
        )
    return hits[0].parent


class _OVSEDWrapper:
    """Wraps a compiled OpenVINO model to match the ort.InferenceSession interface."""
    def __init__(self, compiled):
        self._ov = compiled

    def get_inputs(self):
        class _Input:
            name = "mel"
        return [_Input()]

    def run(self, output_names, feed):
        result = self._ov(feed)
        return [np.asarray(v) for v in result.values()]


def make_sed_session(path):
    if _OV_AVAILABLE:
        try:
            import onnx as _onnx_lib, io as _ov_io
            proto = _onnx_lib.load(str(path))
            for nd in proto.graph.node:
                if nd.op_type == "BatchNormalization":
                    for at in nd.attribute:
                        if at.name == "training_mode" and at.i != 0:
                            at.i = 0
                    while len(nd.output) > 1:
                        nd.output.pop()
            buf = _ov_io.BytesIO()
            _onnx_lib.save(proto, buf)
            core = ov.Core()
            core.set_property("CPU", {"PERFORMANCE_HINT": "LATENCY"})
            compiled = core.compile_model(core.read_model(buf.getvalue()), "CPU")
            print(f"  SED OpenVINO: {Path(path).name}")
            return _OVSEDWrapper(compiled)
        except Exception as _e:
            print(f"  SED OpenVINO failed ({_e}) — using ORT")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path),
        sess_options=so,
        providers=["CPUExecutionProvider"]
    )


import torchaudio.transforms as _T_sed

_sed_mel_tf = _T_sed.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
    n_mels=N_MELS_SED, f_min=FMIN_SED, f_max=FMAX_SED, power=2.0,
)
_sed_atodb = _T_sed.AmplitudeToDB(top_db=TOP_DB_SED)


def audio_to_mel(chunks_np):
    """(N, 160000) float32 → (N, 1, n_mels, T) float32. Vectorised torchaudio."""
    t   = torch.from_numpy(chunks_np)                    # (N, 160000)
    mel = _sed_atodb(_sed_mel_tf(t))                     # (N, n_mels, T)
    mn  = mel.flatten(1).mean(1)[:, None, None]
    std = mel.flatten(1).std(1)[:, None, None]
    mel = (mel - mn) / (std + 1e-6)
    return mel.unsqueeze(1).numpy().astype(np.float32)   # (N, 1, n_mels, T)


def file_to_sed_chunks(path):
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)

    if y.ndim == 2:
        y = y.mean(axis=1)

    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)

    n = 60 * SR

    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]

    chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
    ends   = np.arange(1, N_WINDOWS + 1) * WINDOW_SEC

    return chunks, ends


def sigmoid_sed(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


# Load the 5 SED fold models
sed_dir = find_sed_dir()

sed_fold_paths = sorted(
    sed_dir.glob("sed_fold*.onnx"),
    key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1))
)

sed_sessions = [make_sed_session(p) for p in sed_fold_paths]

print(f"SED dir: {sed_dir}")
print(f"SED folds loaded: {[p.name for p in sed_fold_paths]}")


# Run on the exact same test files used by Cell 9/10
sed_rows, sed_preds = [], []
_t0_sed = time.time()
SED_GAUSSIAN_SMOOTH = True

# Prefetch: overlap OGG decode (releases GIL) with 5× ONNX inference
_SED_PREFETCH = 2
_sed_files = list(test_paths)
_sed_n     = len(_sed_files)

with concurrent.futures.ThreadPoolExecutor(max_workers=_SED_PREFETCH) as _sed_pool:
    _sed_pending = []
    _sed_nxt     = _SED_PREFETCH

    for _f in _sed_files[:_SED_PREFETCH]:
        _sed_pending.append((_f, _sed_pool.submit(file_to_sed_chunks, _f)))

    for i, (_path, _fut) in enumerate(tqdm(_sed_pending, total=_sed_n, desc='SED'), 1):
        if _sed_nxt < _sed_n:
            _sed_pending.append((_sed_files[_sed_nxt], _sed_pool.submit(file_to_sed_chunks, _sed_files[_sed_nxt])))
            _sed_nxt += 1

        chunks, ends = _fut.result()
        mel = audio_to_mel(chunks)

        p_sum = np.zeros((len(chunks), N_CLASSES), dtype=np.float32)

        for sess in sed_sessions:
            outs = sess.run(None, {sess.get_inputs()[0].name: mel})

            clip_logits = outs[0]             # (12, 234)
            frame_max   = outs[1].max(axis=1) # (12, 234)

            p_sum += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)

        p_mean = p_sum / len(sed_sessions)

        if SED_GAUSSIAN_SMOOTH and len(p_mean) > 1:
            p_mean = gaussian_filter1d(
                p_mean,
                sigma=0.65,
                axis=0,
                mode="nearest"
            ).astype(np.float32)

        stem = _path.stem

        sed_rows.extend([f"{stem}_{int(t)}" for t in ends])
        sed_preds.append(p_mean)

        if i == 1 or i % 50 == 0 or i == _sed_n:
            print(f"SED: {i}/{_sed_n} | {time.time()-_t0_sed:.1f}s")


sed_preds_arr = np.concatenate(sed_preds, axis=0)
_PROBS_CACHE["sed"] = (np.array(sed_rows), np.clip(sed_preds_arr, 0.0, 1.0).astype(np.float32))
print(f"SED probs cached  shape={sed_preds_arr.shape}")

del sed_sessions, sed_preds, sed_preds_arr, mel, chunks
del _sed_pending  # free 600 futures × 7.7 MB cached audio = ~4.6 GB
gc.collect()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # r1 NFnet (exp53)
# BCE loss, 20s clips with 5s chunks. 0.919 LB / 0.926 PB

# %% [code] {"jupyter":{"outputs_hidden":false}}
import math
import re as _re
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from tqdm.auto import tqdm

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

warnings.filterwarnings('ignore')
print(f'PyTorch      : {torch.__version__}')
print(f'Torchaudio   : {torchaudio.__version__}')
print(f'OnnxRuntime  : {ort.__version__ if _ORT_AVAILABLE else "not installed"}')
print(f'Device       : CPU (required by competition rules)')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Configuration

# %% [code] {"jupyter":{"outputs_hidden":false}}
class CFG:
    # ── Competition data ───────────────────────────────────────────────────
    BASE_DIR       = Path('/kaggle/input/competitions/birdclef-2026')
    TEST_DIR       = BASE_DIR / 'test_soundscapes'
    SAMPLE_SUB     = BASE_DIR / 'sample_submission.csv'

    # ── Saved artefacts from training notebook ─────────────────────────────
    # Update the dataset slug to match your uploaded dataset name.
    ARTEFACT_DIR   = Path('/kaggle/input/datasets/tennogh/birdclef2026-2nd-place-models/exp53')
    SOUNDSCAPE_DIR = Path('/kaggle/input/competitions/birdclef-2026/train_soundscapes')
    CHECKPOINT     = ARTEFACT_DIR / 'swa_model.pth'    # PyTorch fallback
    ONNX_MODEL     = ARTEFACT_DIR / 'swa_model.onnx'   # preferred; set to None to force PyTorch
    LABEL_MAP      = ARTEFACT_DIR / 'label_map.npy'

    # ── OnnxRuntime thread count ───────────────────────────────────────────
    # Kaggle CPU notebooks have 4 cores; using all of them for ORT intra-op
    # parallelism gives the best single-session throughput.
    ORT_THREADS    = 4

    OUTPUT_DIR     = Path('/kaggle/working')

    # ── Audio — must match training config exactly ─────────────────────────
    SR             = 32000
    CHUNK_DURATION = 5        # seconds per inference chunk
    N_FFT          = 4096
    HOP_LENGTH     = 512
    N_MELS         = 224
    FMIN           = 0
    FMAX           = 16000

    # ── Model — must match training config exactly ─────────────────────────
    MODEL_NAME     = 'convnext_base.clip_laion2b_augreg_ft_in1k'
    DROP_PATH_RATE = 0.15
    USE_GEM        = False    # set True if trained with --use_gem true
    GEM_P_INIT     = 3.0     # must match gem_p_init used during training
    IMAGENET_NORM  = False    # set True for ViT models trained with imagenet_norm=True

    # ── Temperature scaling ────────────────────────────────────────────────
    TAXONOMY_CSV       = BASE_DIR / 'taxonomy.csv'
    TAXON_TEMPERATURES = {'Aves': 1.0, 'Insecta': 1.0, 'Amphibia': 1.0}

    # ── Post-processing ────────────────────────────────────────────────────
    # Temporal TTA: re-run inference with the full waveform circularly rolled
    # by ±TTA_SHIFT samples, then average with the unshifted predictions.
    # Weight: (1 * orig + 2 * roll_fwd + 2 * roll_bwd) / 5
    # ±40,000 samples = ±1.25s at SR=32000. Zero training cost.
    TEMPORAL_TTA       = False
    TTA_SHIFT          = 40_000   # samples

    # Temporal smoothing: [0.20, 0.60, 0.20] kernel. Disabled — 0 to -0.001 LB.
    TEMPORAL_SMOOTHING = False

    # Delta-shift smoothing: gentle temporal smoothing across consecutive chunks.
    # new[t] = (1 - alpha) * old[t] + 0.5 * alpha * (old[t-1] + old[t+1])
    # OOF isolation tests: +0.003 to +0.007 AUC.
    DELTA_SMOOTH       = False
    DELTA_SMOOTH_ALPHA = 0.15

    # Rank-aware scaling: multiply each chunk's predictions by file_max^power
    # per species. Suppresses species the model is never confident about in
    # a given file. OOF isolation tests: +0.002 to +0.017 AUC.
    RANK_AWARE         = False
    RANK_AWARE_POWER   = 0.4

    # Confidence scaling: multiply each chunk by the per-species top-K mean
    # across all chunks of the file. Validated: +0.018–0.027 LB.
    CONF_SCALE         = True
    CONF_SCALE_TOP_K   = 2

    # Diel prior: class-level likelihood ratios from training soundscapes.
    # Disabled — causes -0.006 LB (double-counts soundscape training signal).
    DIEL_PRIOR         = False

    # ── Inference ─────────────────────────────────────────────────────────
    DEVICE     = 'cpu'    # competition constraint
    BATCH_SIZE = 12       # all chunks of one 1-min file in one pass

    # ── I/O prefetch ──────────────────────────────────────────────────────
    # Number of files to load ahead of the main thread in a background thread.
    # torchaudio.load + OGG decode releases the GIL, so this overlaps disk
    # I/O and audio decoding with ONNX inference at no extra CPU contention.
    # 2 is enough to hide latency; more wastes RAM (each file ~7 MB as float32).
    PREFETCH   = 2

    # ── Derived ────────────────────────────────────────────────────────────
    @classmethod
    def chunk_frames(cls):
        """Number of mel time frames for one CHUNK_DURATION clip."""
        return math.floor(cls.SR * cls.CHUNK_DURATION / cls.HOP_LENGTH) + 1

CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Chunk frames : {CFG.chunk_frames()}  (mel T for {CFG.CHUNK_DURATION}s at hop={CFG.HOP_LENGTH})')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Load Submission Template & Label Map

# %% [code] {"jupyter":{"outputs_hidden":false}}
sample_sub      = pd.read_csv(CFG.SAMPLE_SUB)
SPECIES_COLS    = [c for c in sample_sub.columns if c != 'row_id']
NUM_SUB_SPECIES = len(SPECIES_COLS)
print(f'Submission species : {NUM_SUB_SPECIES}')
print(f'Example row_id     : {sample_sub["row_id"].iloc[0]}')
sample_sub.head(3)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# label_map.npy is saved as a dict {int -> species_str} via np.save.
# np.load returns a 0-d object array; .item() extracts the dict.
label_map   = np.load(CFG.LABEL_MAP, allow_pickle=True).item()
NUM_CLASSES = len(label_map)
idx2species = label_map    # {int: str}

UNIFORM_PRIOR = 1.0 / NUM_SUB_SPECIES

# model output index -> submission column index (-1 = not in submission)
model_to_sub = np.full(NUM_CLASSES, -1, dtype=np.int32)
sub_col_map  = {sp: j for j, sp in enumerate(SPECIES_COLS)}
for model_idx, sp in idx2species.items():
    if sp in sub_col_map:
        model_to_sub[model_idx] = sub_col_map[sp]

n_mapped = (model_to_sub >= 0).sum()
_model_mapped = model_to_sub >= 0  # bool mask for vectorised scatter
print(f'Model classes                        : {NUM_CLASSES}')
print(f'Mapped to submission columns         : {n_mapped}')
print(f'Submission columns on uniform prior  : {NUM_SUB_SPECIES - n_mapped}')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Load Model
# #
# Tries to load an ONNX model via OnnxRuntime (faster on CPU).
# Falls back to the inline PyTorch BirdCLEFModel if the ONNX file is absent
# or onnxruntime is not installed.
# #
# Inline PyTorch class matches `Codebase/birdclef/model.py::BirdCLEFModel`
# and supports CNN + ViT backbones and optional GeM pooling.
# Must stay in sync with the codebase version.

# %% [code] {"jupyter":{"outputs_hidden":false}}
_VIT_PREFIXES = ('vit_', 'deit_', 'beit_', 'eva_')


def _is_vit(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _VIT_PREFIXES)


class GeMPool1d(nn.Module):
    """GeM pooling over dim=2 of (B, C, H, W). p loaded from checkpoint."""
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class BirdCLEFModel(nn.Module):
    """
    SED architecture matching Codebase/birdclef/model.py::BirdCLEFModel.
    Only used as PyTorch fallback when ONNX is unavailable.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        n_mels: int,
        chunk_frames: int,
        drop_path_rate: float = 0.0,
        use_gem: bool = False,
        gem_p_init: float = 3.0,
    ):
        super().__init__()
        self._is_vit  = _is_vit(model_name)
        self.num_classes = num_classes
        self._use_gem = use_gem
        if use_gem:
            self.gem_pool = GeMPool1d(p_init=gem_p_init)

        if self._is_vit:
            self.backbone = timm.create_model(
                model_name, pretrained=False, num_classes=0, global_pool='',
                in_chans=3, img_size=(n_mels, chunk_frames),
                drop_path_rate=drop_path_rate, dynamic_img_size=True,
            )
            with torch.no_grad():
                dummy      = torch.zeros(1, 3, n_mels, chunk_frames)
                tokens     = self.backbone(dummy)[:, 1:, :]
                n_features = tokens.shape[2]
                self.h_patches = n_mels // self.backbone.patch_embed.patch_size[0]
        else:
            self.backbone = timm.create_model(
                model_name, pretrained=False, num_classes=0, global_pool='',
                in_chans=3, drop_path_rate=drop_path_rate,
            )
            with torch.no_grad():
                dummy      = torch.zeros(1, 3, n_mels, chunk_frames)
                n_features = self.backbone(dummy).shape[1]

        self.fc      = nn.Linear(n_features, num_classes)
        self.att_fc  = nn.Linear(n_features, num_classes)
        self.bn      = nn.BatchNorm1d(n_features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor):
        if self._is_vit:
            tokens = self.backbone(x)[:, 1:, :]              # (B, N, D)
            B, N, D = tokens.shape
            w_patches = N // self.h_patches
            feat = tokens.reshape(B, self.h_patches, w_patches, D).permute(0, 3, 1, 2)
            feat = self.gem_pool(feat) if self._use_gem else feat.mean(dim=2)
            feat = feat.permute(0, 2, 1)                     # (B, T, D)
        else:
            feat = self.backbone(x)                          # (B, C, H, W)
            feat = self.gem_pool(feat) if self._use_gem else feat.mean(dim=2)
            feat = feat.permute(0, 2, 1)                     # (B, T, C)

        B, T, C = feat.shape
        feat            = self.bn(feat.reshape(B * T, C)).reshape(B, T, C)
        feat            = self.dropout(feat)
        frame_logits    = self.fc(feat)
        att_weights     = torch.softmax(self.att_fc(feat), dim=1)
        clipwise_logits = frame_logits.mean(dim=1)
        att_clipwise    = (frame_logits * att_weights).sum(dim=1)
        return clipwise_logits, att_clipwise, frame_logits


# ── Backend selection: OpenVINO > OnnxRuntime > PyTorch ──────────────────────
_use_ov_nfnet = False
_ov_nfnet     = None
_use_onnx     = False
_ort_session  = None
model         = None

_onnx_path = CFG.ONNX_MODEL
if _OV_AVAILABLE and _onnx_path is not None and Path(_onnx_path).exists():
    try:
        _ov_core = ov.Core()
        _ov_core.set_property("CPU", {"PERFORMANCE_HINT": "LATENCY"})
        # Some ONNX exports set BatchNormalization training_mode=1 (opset-17 artefact).
        # OV rejects it; patch to inference mode (training_mode=0, 1 output) in memory.
        try:
            import onnx as _onnx_lib, io as _ov_io
            _proto   = _onnx_lib.load(str(_onnx_path))
            _patched = False
            for _nd in _proto.graph.node:
                if _nd.op_type == 'BatchNormalization':
                    for _at in _nd.attribute:
                        if _at.name == 'training_mode' and _at.i != 0:
                            _at.i    = 0
                            _patched = True
                    while len(_nd.output) > 1:
                        _nd.output.pop()
                        _patched = True
            if _patched:
                print('  Patched BatchNormalization training_mode → 0')
            _buf = _ov_io.BytesIO()
            _onnx_lib.save(_proto, _buf)
            _ov_nfnet = _ov_core.compile_model(_ov_core.read_model(_buf.getvalue()), "CPU")
        except ImportError:
            # onnx package not available — load directly (may still fail if BN is training-mode)
            _ov_nfnet = _ov_core.compile_model(_ov_core.read_model(str(_onnx_path)), "CPU")
        _use_ov_nfnet = True
        print(f'Backend       : OpenVINO  ({_onnx_path.name})')
    except Exception as _ov_exc:
        print(f'OpenVINO load failed ({_ov_exc}) — trying OnnxRuntime')

if not _use_ov_nfnet and _ORT_AVAILABLE and _onnx_path is not None and Path(_onnx_path).exists():
    _sess_opts = ort.SessionOptions()
    _sess_opts.intra_op_num_threads      = CFG.ORT_THREADS
    _sess_opts.graph_optimization_level  = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _ort_session = ort.InferenceSession(
        str(_onnx_path),
        sess_options=_sess_opts,
        providers=['CPUExecutionProvider'],
    )
    _use_onnx = True
    print(f'Backend       : OnnxRuntime  ({_onnx_path.name})')
    print(f'ORT threads   : {CFG.ORT_THREADS}')

if not _use_ov_nfnet and not _use_onnx:
    if not _ORT_AVAILABLE:
        print('Backend       : PyTorch  (onnxruntime not installed)')
    elif _onnx_path is None:
        print('Backend       : PyTorch  (CFG.ONNX_MODEL is None)')
    else:
        print(f'Backend       : PyTorch  ({_onnx_path.name} not found)')
    model = BirdCLEFModel(
        model_name     = CFG.MODEL_NAME,
        num_classes    = NUM_CLASSES,
        n_mels         = CFG.N_MELS,
        chunk_frames   = CFG.chunk_frames(),
        drop_path_rate = CFG.DROP_PATH_RATE,
        use_gem        = CFG.USE_GEM,
        gem_p_init     = CFG.GEM_P_INIT,
    )
    model.load_state_dict(torch.load(CFG.CHECKPOINT, map_location='cpu'))
    model.eval()

print(f'Model classes : {NUM_CLASSES}')
print(f'GeM pooling   : {CFG.USE_GEM}')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4b. Temperature Scaling Setup
# #
# Builds a per-species temperature vector in model-output order.
# Applied as `sigmoid(logits / temp)` in `_forward_probs`.
# All values currently 1.0 (no scaling). Edit CFG.TAXON_TEMPERATURES to tune.

# %% [code] {"jupyter":{"outputs_hidden":false}}
taxonomy  = pd.read_csv(CFG.TAXONOMY_CSV)
taxon_map = taxonomy.set_index('primary_label')['class_name'].to_dict()
taxon_map = {str(k): v for k, v in taxon_map.items()}

model_temp = torch.ones(NUM_CLASSES)
for model_idx, sp in idx2species.items():
    cls = taxon_map.get(str(sp), 'Aves')
    model_temp[model_idx] = CFG.TAXON_TEMPERATURES.get(cls, 1.0)

temp_counts = Counter(model_temp.tolist())
print('Temperature vector built:')
for t, n in sorted(temp_counts.items()):
    print(f'  T={t:.2f}  →  {n} species')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4c. Diel Prior Setup
# #
# Class-level likelihood ratios LR(class | hour) derived from 739 labelled
# 5-second chunks across 66 training soundscape files.
# Disabled by default — causes -0.006 LB (double-counts soundscape training signal).

# %% [code] {"jupyter":{"outputs_hidden":false}}
_DIEL_LR = {
    'Amphibia': {0: 0.776, 1: 1.118, 2: 1.274, 3: 0.333, 4: 0.441,
                 6: 0.100, 7: 0.100, 18: 1.256, 19: 1.289,
                 20: 1.299, 21: 1.300, 22: 1.297, 23: 1.300},
    'Aves':     {0: 1.281, 1: 1.440, 2: 0.942, 3: 0.746, 4: 1.741,
                 6: 3.447, 7: 1.647, 18: 3.348, 19: 0.141,
                 20: 0.102, 21: 0.100, 22: 0.100, 23: 1.549},
    'Insecta':  {0: 0.100, 1: 0.100, 2: 0.100, 3: 3.271, 4: 2.908,
                 6: 4.347, 7: 4.332, 18: 0.169, 19: 2.908,
                 20: 0.100, 21: 0.100, 22: 0.100, 23: 0.100},
}
_HOUR_RE = _re.compile(r'_(\d{8})_(\d{6})')

_diel_log_lr_by_hour = {}
for _h in range(24):
    _log_lr = np.zeros(NUM_CLASSES, dtype=np.float32)
    for _model_idx, _sp in idx2species.items():
        _cls = taxon_map.get(str(_sp))
        if _cls in _DIEL_LR:
            _lr = _DIEL_LR[_cls].get(_h, 1.0)
            _log_lr[_model_idx] = np.log(_lr)
    _diel_log_lr_by_hour[_h] = _log_lr


def apply_diel_prior(probs: np.ndarray, filename: str) -> np.ndarray:
    """
    Adjust (N_chunks, NUM_CLASSES) probabilities by class-level diel LR.
    Extracts hour from filename, applies log-LR shift in logit space.
    Returns probs unchanged if hour cannot be parsed.
    """
    m = _HOUR_RE.search(filename)
    if m is None:
        return probs
    hour   = int(m.group(2)[:2])
    log_lr = _diel_log_lr_by_hour.get(hour, _diel_log_lr_by_hour[0] * 0)
    if not np.any(log_lr):
        return probs
    eps    = 1e-6
    logits = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))
    return (1.0 / (1.0 + np.exp(-(logits + log_lr[np.newaxis, :])))).astype(np.float32)

_test_hour  = 6
_test_probs = np.full((1, NUM_CLASSES), 0.3, dtype=np.float32)
_adjusted   = apply_diel_prior(_test_probs, f'BC2026_Test_0001_S05_20250101_{_test_hour:02d}0000.ogg')
print(f'Diel prior sanity check at hour {_test_hour}:')
for _sp, _cls in [('rufhor2', 'Aves'), ('22985', 'Amphibia'), ('1161364', 'Insecta')]:
    _idx = next((i for i, s in idx2species.items() if str(s) == _sp), None)
    if _idx is not None:
        print(f'  {_sp} ({_cls}): {_test_probs[0, _idx]:.3f} → {_adjusted[0, _idx]:.3f}  '
              f'(LR={_DIEL_LR.get(_cls, {}).get(_test_hour, 1.0):.3f})')
print('Diel prior setup complete.')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Audio Utilities

# %% [code] {"jupyter":{"outputs_hidden":false}}
mel_transform   = T.MelSpectrogram(
    sample_rate = CFG.SR,
    n_fft       = CFG.N_FFT,
    hop_length  = CFG.HOP_LENGTH,
    n_mels      = CFG.N_MELS,
    f_min       = CFG.FMIN,
    f_max       = CFG.FMAX,
)
amplitude_to_db = T.AmplitudeToDB(top_db=80)

# ImageNet normalisation constants — only applied when CFG.IMAGENET_NORM is True
# (ViT models trained with imagenet_norm=True in the training config).
_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_and_chunk(filepath: Path) -> tuple[torch.Tensor, int]:
    """
    Load a soundscape, resample if needed, split into non-overlapping
    CHUNK_DURATION-second chunks.

    Returns:
        chunks   : (N_chunks, chunk_len) float32 tensor
        n_chunks : number of chunks
    """
    waveform, orig_sr = torchaudio.load(str(filepath))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != CFG.SR:
        waveform = torchaudio.functional.resample(waveform, orig_sr, CFG.SR)

    chunk_len = CFG.SR * CFG.CHUNK_DURATION
    total_len = waveform.shape[-1]
    remainder = total_len % chunk_len
    if remainder:
        waveform = F.pad(waveform, (0, chunk_len - remainder))

    chunks = waveform.squeeze(0).reshape(-1, chunk_len)
    return chunks, chunks.shape[0]


def chunks_to_mel_batch(chunks: torch.Tensor) -> torch.Tensor:
    """
    Convert (N_chunks, chunk_len) → (N_chunks, 3, N_MELS, T_frames).
    Vectorised; no Python loop per chunk.
    Applies ImageNet normalisation when CFG.IMAGENET_NORM is True.
    """
    mel = mel_transform(chunks)                           # (N, n_mels, T)
    mel = amplitude_to_db(mel)
    mn  = mel.flatten(1).min(dim=1).values[:, None, None]
    mx  = mel.flatten(1).max(dim=1).values[:, None, None]
    mel = (mel - mn) / (mx - mn + 1e-6)
    mel = mel.unsqueeze(1).repeat(1, 3, 1, 1)            # (N, 3, n_mels, T)
    if CFG.IMAGENET_NORM:
        mel = (mel - _IN_MEAN) / _IN_STD
    return mel

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 6. Inference Loop

# %% [code] {"jupyter":{"outputs_hidden":false}}
_model_temp_np = None   # lazily set to model_temp.numpy() on first non-PyTorch call


def _forward_probs(batch: torch.Tensor) -> np.ndarray:
    """
    Forward pass → probabilities (N_chunks, NUM_CLASSES).
    Priority: OpenVINO > OnnxRuntime > PyTorch.
    Temperature scaling is applied in the same dtype as the backend output.
    """
    global _model_temp_np
    if _use_ov_nfnet:
        if _model_temp_np is None:
            _model_temp_np = model_temp.numpy()
        logits = list(_ov_nfnet({"mel_spec": batch.numpy()}).values())[0]
        return (1.0 / (1.0 + np.exp(-logits / _model_temp_np))).astype(np.float32)
    elif _use_onnx:
        if _model_temp_np is None:
            _model_temp_np = model_temp.numpy()
        logits = _ort_session.run(None, {'mel_spec': batch.numpy()})[0]  # (N, C) float32
        return (1.0 / (1.0 + np.exp(-logits / _model_temp_np))).astype(np.float32)
    else:
        with torch.inference_mode():
            _, att_clipwise, _ = model(batch)
            return torch.sigmoid(att_clipwise / model_temp).numpy()


def _roll_chunks(chunks: torch.Tensor, shift: int) -> torch.Tensor:
    """Circularly roll the full waveform by `shift` samples, then re-chunk."""
    flat     = chunks.reshape(-1)
    flat     = torch.roll(flat, shifts=shift)
    n_chunks = chunks.shape[0]
    return flat.reshape(n_chunks, -1)


def _process_chunks(filepath: Path, chunks: torch.Tensor, n_chunks: int) -> pd.DataFrame:
    """
    Compute-only part of inference: mel → ONNX → post-processing → DataFrame.
    Separated from I/O so the main loop can overlap loading with inference.

    Post-processing order:
      1. Temporal TTA        — average over ±1.25s circular rolls (1:2:2 weights)
      2. Diel prior          — class-level LR in logit space (disabled: -0.006 LB)
      3. Temporal smoothing  — [0.20, 0.60, 0.20] kernel (disabled: 0 to -0.001 LB)
      4. Delta-shift smooth  — gentle [0.075, 0.85, 0.075] kernel (+0.003 to +0.007)
      5. Rank-aware scaling  — multiply by file_max^power per species (+0.002 to +0.017)
      6. Confidence scaling  — multiply by per-species top-K mean (+0.018–0.027 LB)
    """
    stem  = filepath.stem

    probs = _forward_probs(chunks_to_mel_batch(chunks))   # (N, NUM_CLASSES)

    if CFG.TEMPORAL_TTA:
        probs_fwd = _forward_probs(chunks_to_mel_batch(_roll_chunks(chunks, +CFG.TTA_SHIFT)))
        probs_bwd = _forward_probs(chunks_to_mel_batch(_roll_chunks(chunks, -CFG.TTA_SHIFT)))
        probs = (probs + 2.0 * probs_fwd + 2.0 * probs_bwd) / 5.0

    if CFG.DIEL_PRIOR:
        probs = apply_diel_prior(probs, filepath.name)

    if CFG.TEMPORAL_SMOOTHING:
        padded = np.concatenate([probs[:1], probs, probs[-1:]], axis=0)
        probs  = 0.20 * padded[:-2] + 0.60 * padded[1:-1] + 0.20 * padded[2:]

    if CFG.DELTA_SMOOTH:
        a     = CFG.DELTA_SMOOTH_ALPHA
        prev  = np.concatenate([probs[:1],  probs[:-1]], axis=0)
        nxt   = np.concatenate([probs[1:],  probs[-1:]], axis=0)
        probs = (1 - a) * probs + 0.5 * a * (prev + nxt)

    if CFG.RANK_AWARE:
        file_max = probs.max(axis=0)
        scale    = np.power(np.clip(file_max, 1e-9, 1.0), CFG.RANK_AWARE_POWER)
        probs    = probs * scale[None, :]

    if CFG.CONF_SCALE:
        top_k_mean = np.sort(probs, axis=0)[-CFG.CONF_SCALE_TOP_K:].mean(axis=0)
        probs      = probs * top_k_mean[None, :]

    out = np.full((n_chunks, NUM_SUB_SPECIES), UNIFORM_PRIOR, dtype=np.float32)
    out[:, model_to_sub[_model_mapped]] = probs[:, _model_mapped]

    row_ids = [f'{stem}_{(i + 1) * CFG.CHUNK_DURATION}' for i in range(n_chunks)]
    df = pd.DataFrame(out, columns=SPECIES_COLS)
    df.insert(0, 'row_id', row_ids)
    return df


# ── Main inference loop ────────────────────────────────────────────────────
test_files = sorted(CFG.TEST_DIR.glob('*.ogg'))
print(f'Test files found: {len(test_files)}')

if len(test_files) == 0:
    print('Hidden test not mounted. Dry-run on first 20 train soundscapes.')
    test_files = sorted(CFG.SOUNDSCAPE_DIR.glob('*.ogg'))[:n_dry_run]

all_preds = []
t0        = time.time()

# Prefetch pipeline: a thread pool keeps PREFETCH load_and_chunk futures in
# flight while the main thread runs mel + ONNX + post-processing.
# torchaudio.load and OGG decoding release the GIL, so this overlaps disk
# I/O with compute without competing with ORT's CPU threads.
with ThreadPoolExecutor(max_workers=CFG.PREFETCH) as pool:
    # pending is a list of (filepath, future); we append to it while iterating,
    # which is valid in Python — the for-loop will see newly appended items.
    pending  = []
    files    = list(test_files)
    n_files  = len(files)
    next_idx = 0

    # Seed: submit first PREFETCH files
    for f in files[:CFG.PREFETCH]:
        pending.append((f, pool.submit(load_and_chunk, f)))
    next_idx = CFG.PREFETCH

    for filepath, future in tqdm(pending, total=n_files, desc='Inference'):
        # Submit the next file immediately so it loads while we do inference
        if next_idx < n_files:
            pending.append((files[next_idx], pool.submit(load_and_chunk, files[next_idx])))
            next_idx += 1

        chunks, n_chunks = future.result()
        all_preds.append(_process_chunks(filepath, chunks, n_chunks))

elapsed  = time.time() - t0
n_passes = 3 if CFG.TEMPORAL_TTA else 1
print(f'\nDone in {elapsed / 60:.1f} min  ({elapsed / len(test_files):.2f} s/file, {n_passes} pass(es)/file)')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 7. Build & Validate Submission

# %% [code] {"jupyter":{"outputs_hidden":false}}
preds = pd.concat(all_preds, ignore_index=True)

submission = preds
for sp in SPECIES_COLS:
    submission[sp] = submission[sp].fillna(UNIFORM_PRIOR)

print(f'Submission shape  : {submission.shape}')
print(f'Rows              : {submission.shape[0]}')
print(f'Species columns   : {submission.shape[1] - 1}')
print(f'All checks passed ✓')
submission.head(3)

# %% [code] {"jupyter":{"outputs_hidden":false}}
_PROBS_CACHE["exp53"] = (submission["row_id"].values.copy(), submission[SPECIES_COLS].to_numpy(np.float32))
print(f"exp53 probs cached  shape={submission.shape}")

# Free NFNet model before ensemble — ConvNeXt-base is ~350 MB
if _use_ov_nfnet and _ov_nfnet is not None:
    del _ov_nfnet
elif _use_onnx and _ort_session is not None:
    del _ort_session
elif model is not None:
    del model
import gc as _gc; _gc.collect()

# %% [markdown]
# # r3 v2_s + r4 v2_s + r5 NFnet (exp72e+exp79b+exp88)
# exp72e: v2_s with BCE loss (0.933 LB 0.928 PB)
# 
# exp79b: v2_s with soft AUC loss + 0.25 BCE loss (0.938 LB 0.935 PB)
# 
# exp88: NFnet with soft AUC loss + 0.25 BCE loss (0.933 LB 0.932 PB)
# 
# All 3 share the same mel specs, so inference is done within the same loop. Named "Effnet section" but really the common point is the mel specs.

# %% [code] {"jupyter":{"outputs_hidden":false}}
# exp72e + exp79b (+ optional exp88) share a single file-loop.
# Mel specs are identical across all three: N_FFT=2048, HOP=512, N_MELS=128.
# Each file is loaded once; mel computed once; all ONNX sessions run
# back-to-back on the same batch; then each model's CSV is written.

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from tqdm.auto import tqdm

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

try:
    import openvino as ov
    _OV_AVAILABLE = True
except ImportError:
    _OV_AVAILABLE = False

warnings.filterwarnings('ignore')

# ── Shared mel config (must match all models' training config) ────────────
_SR               = 32000
_CHUNK_DURATION   = 5
_N_FFT            = 2048
_HOP_LENGTH       = 512
_N_MELS           = 128
_FMIN             = 20
_FMAX             = 16000
_ORT_THREADS      = 4
_PREFETCH         = 2
_CONF_SCALE_TOP_K = 2

_BASE_DIR       = Path('/kaggle/input/competitions/birdclef-2026')
_TEST_DIR       = _BASE_DIR / 'test_soundscapes'
_SOUNDSCAPE_DIR = _BASE_DIR / 'train_soundscapes'
_OUTPUT_DIR     = Path('/kaggle/working')
_CHUNK_LEN      = _SR * _CHUNK_DURATION

# ── Model registry ────────────────────────────────────────────────────────
# To test exp88: uncomment the last entry and adjust the dataset slug.
_EFFNET_MODELS = [
    {
        "name":    "exp72e",
        "onnx":    Path('/kaggle/input/datasets/tennogh/birdclef2026-2nd-place-models/exp72e/best_model.onnx'),
        "lmap":    Path('/kaggle/input/datasets/tennogh/birdclef2026-2nd-place-models/exp72e/label_map.npy'),
        "csv_out": "submission_exp72e.csv",
    },
    {
        "name":    "exp79b",
        "onnx":    Path('/kaggle/input/datasets/tennogh/birdclef2026-2nd-place-models/exp79b/best_model.onnx'),
        "lmap":    Path('/kaggle/input/datasets/tennogh/birdclef2026-2nd-place-models/exp79b/label_map.npy'),
        "csv_out": "submission_exp79b.csv",
    },
     {
         "name":    "exp88",
         "onnx":    Path('/kaggle/input/datasets/tennogh/birdclef2026-2nd-place-models/exp88/swa_model.onnx'),
         "lmap":    Path('/kaggle/input/datasets/tennogh/birdclef2026-2nd-place-models/exp88/label_map.npy'),
         "csv_out": "submission_exp88.csv",
     },
]

# ── Submission template (load once) ──────────────────────────────────────
_sample_sub      = pd.read_csv(_BASE_DIR / 'sample_submission.csv')
_SPECIES_COLS    = [c for c in _sample_sub.columns if c != 'row_id']
_NUM_SUB_SPECIES = len(_SPECIES_COLS)
_UNIFORM_PRIOR   = 1.0 / _NUM_SUB_SPECIES
_sub_col_map     = {sp: j for j, sp in enumerate(_SPECIES_COLS)}

# ── Load each model's ONNX session + build label→submission mapping ───────
def _make_session(onnx_path):
    """OpenVINO (preferred) → OnnxRuntime fallback. Consistent variable names."""
    if _OV_AVAILABLE:
        try:
            import onnx as _ol, io as _oi
            proto   = _ol.load(str(onnx_path))
            patched = False
            for nd in proto.graph.node:
                if nd.op_type == 'BatchNormalization':
                    for at in nd.attribute:
                        if at.name == 'training_mode' and at.i != 0:
                            at.i = 0; patched = True
                    while len(nd.output) > 1:
                        nd.output.pop(); patched = True
            if patched:
                print('    Patched BN training_mode → 0')
            buf = _oi.BytesIO(); _ol.save(proto, buf)
            core     = ov.Core()
            core.set_property('CPU', {'PERFORMANCE_HINT': 'LATENCY'})
            compiled = core.compile_model(core.read_model(buf.getvalue()), 'CPU')
            return 'ov', compiled, compiled.output(0)
        except Exception as e:
            print(f'    OV failed ({e}) → ORT')
    if _ORT_AVAILABLE:
        so = ort.SessionOptions()
        so.intra_op_num_threads     = _ORT_THREADS
        so.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                    providers=['CPUExecutionProvider'])
        return 'ort', sess, None
    raise RuntimeError(f'No inference backend available for {onnx_path}')

for _md in _EFFNET_MODELS:
    print(f"Loading {_md['name']} ({_md['onnx'].name})…")
    _md['backend'], _md['sess'], _md['ov_out'] = _make_session(_md['onnx'])
    _lm  = np.load(_md['lmap'], allow_pickle=True).item()
    _md['n_cls'] = len(_lm)
    _m2s = np.full(len(_lm), -1, dtype=np.int32)
    for _i, _sp in _lm.items():
        if _sp in _sub_col_map:
            _m2s[_i] = _sub_col_map[_sp]
    _md['m2s']    = _m2s
    _md['mapped'] = _m2s >= 0
    _md['preds']  = []
    print(f"  backend={_md['backend']}  classes={_md['n_cls']}  mapped={_md['mapped'].sum()}")

# ── Shared mel transform ──────────────────────────────────────────────────
_mel_tf = T.MelSpectrogram(sample_rate=_SR, n_fft=_N_FFT, hop_length=_HOP_LENGTH,
                            n_mels=_N_MELS, f_min=_FMIN, f_max=_FMAX)
_atodb  = T.AmplitudeToDB(top_db=80)


def _load_and_chunk(filepath):
    wav, sr = torchaudio.load(str(filepath))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != _SR:
        wav = torchaudio.functional.resample(wav, sr, _SR)
    rem = wav.shape[-1] % _CHUNK_LEN
    if rem:
        wav = F.pad(wav, (0, _CHUNK_LEN - rem))
    chunks = wav.squeeze(0).reshape(-1, _CHUNK_LEN)
    return chunks, chunks.shape[0]


def _to_mel_batch(chunks):
    """(N, chunk_len) → (N, 3, n_mels, T) float32 numpy array — computed once per file."""
    mel = _mel_tf(chunks)
    mel = _atodb(mel)
    mn  = mel.flatten(1).min(dim=1).values[:, None, None]
    mx  = mel.flatten(1).max(dim=1).values[:, None, None]
    mel = (mel - mn) / (mx - mn + 1e-6)
    return mel.unsqueeze(1).repeat(1, 3, 1, 1).numpy().astype(np.float32)


def _run_model(md, batch_np):
    if md['backend'] == 'ov':
        return np.asarray(md['sess']({'mel_spec': batch_np})[md['ov_out']])
    return md['sess'].run(None, {'mel_spec': batch_np})[0]


def _proc_file(filepath, chunks, n_chunks):
    """Mel once → all models sequentially → accumulate predictions."""
    batch_np = _to_mel_batch(chunks)
    stem     = filepath.stem
    row_ids  = [f'{stem}_{(i+1)*_CHUNK_DURATION}' for i in range(n_chunks)]
    for md in _EFFNET_MODELS:
        logits = _run_model(md, batch_np)
        probs  = (1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))).astype(np.float32)
        # Confidence scaling: multiply each chunk by per-species top-2 mean across file
        top_k_mean = np.sort(probs, axis=0)[-_CONF_SCALE_TOP_K:].mean(axis=0)
        probs     *= top_k_mean[None, :]
        out        = np.full((n_chunks, _NUM_SUB_SPECIES), _UNIFORM_PRIOR, dtype=np.float32)
        out[:, md['m2s'][md['mapped']]] = probs[:, md['mapped']]
        df         = pd.DataFrame(out, columns=_SPECIES_COLS)
        df.insert(0, 'row_id', row_ids)
        md['preds'].append(df)


# ── Main loop ─────────────────────────────────────────────────────────────
_test_files = sorted(_TEST_DIR.glob('*.ogg'))
if not _test_files:
    print('Dry-run: using train soundscapes.')
    _test_files = sorted(_SOUNDSCAPE_DIR.glob('*.ogg'))[:n_dry_run]
print(f'Test files: {len(_test_files)}  |  models: {[m["name"] for m in _EFFNET_MODELS]}')

_t0_shared = time.time()
with ThreadPoolExecutor(max_workers=_PREFETCH) as _pool:
    _pending, _files, _nxt = [], list(_test_files), 0
    for _f in _files[:_PREFETCH]:
        _pending.append((_f, _pool.submit(_load_and_chunk, _f)))
    _nxt = _PREFETCH
    for _fp, _fut in tqdm(_pending, total=len(_files), desc='EffNet shared'):
        if _nxt < len(_files):
            _pending.append((_files[_nxt], _pool.submit(_load_and_chunk, _files[_nxt])))
            _nxt += 1
        _chunks, _n_chunks = _fut.result()
        _proc_file(_fp, _chunks, _n_chunks)

_elapsed = time.time() - _t0_shared
print(f'Done in {_elapsed/60:.1f} min  ({_elapsed/len(_test_files):.2f}s/file × {len(_EFFNET_MODELS)} models)')

# ── Cache per-model probs in memory (skip CSV write) ─────────────────────
for _md in _EFFNET_MODELS:
    _sub = pd.concat(_md['preds'], ignore_index=True)
    for _sp in _SPECIES_COLS:
        _sub[_sp] = _sub[_sp].fillna(_UNIFORM_PRIOR)
    _PROBS_CACHE[_md["name"]] = (_sub["row_id"].values.copy(), _sub[_SPECIES_COLS].to_numpy(np.float32))
    print(f"Cached {_md['name']}  shape={_sub.shape}")
    del _sub

# ── Cleanup ───────────────────────────────────────────────────────────────
for _md in _EFFNET_MODELS:
    _md.pop('sess', None)
    _md['preds'].clear()
del _mel_tf, _atodb, _EFFNET_MODELS
del _pending  # free 600 futures × 7.7 MB cached audio = ~4.6 GB
gc.collect()
print('Shared EffNet section complete.')

# %% [markdown]
# # Insecta specialist effnet_b0 (exp85)
# Used only for insecta inference. 

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Imports

# %% [code] {"jupyter":{"outputs_hidden":false}}
import math
import re as _re
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from tqdm.auto import tqdm

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

warnings.filterwarnings('ignore')
print(f'PyTorch      : {torch.__version__}')
print(f'Torchaudio   : {torchaudio.__version__}')
print(f'OnnxRuntime  : {ort.__version__ if _ORT_AVAILABLE else "not installed"}')
print(f'Device       : CPU (required by competition rules)')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Configuration

# %% [code] {"jupyter":{"outputs_hidden":false}}
class CFG:
    # ── Competition data ───────────────────────────────────────────────────
    BASE_DIR       = Path('/kaggle/input/competitions/birdclef-2026')
    TEST_DIR       = BASE_DIR / 'test_soundscapes'
    SAMPLE_SUB     = BASE_DIR / 'sample_submission.csv'

    # ── Saved artefacts from training notebook ─────────────────────────────
    # Update the dataset slug to match your uploaded dataset name.
    ARTEFACT_DIR   = Path('/kaggle/input/datasets/tennogh/birdclef2026-2nd-place-models/exp85')
    SOUNDSCAPE_DIR = Path('/kaggle/input/competitions/birdclef-2026/train_soundscapes')
    CHECKPOINT     = ARTEFACT_DIR / 'swa_model.pth'    # PyTorch fallback
    # ONNX_MODEL is auto-set below after TRAIN_DURATION is defined.
    # Override here to force a specific file, or set to None to force PyTorch.
    ONNX_MODEL     = None   # placeholder — replaced after class definition
    LABEL_MAP      = ARTEFACT_DIR / 'label_map.npy'

    # ── OnnxRuntime thread count ───────────────────────────────────────────
    # Kaggle CPU notebooks have 4 cores; using all of them for ORT intra-op
    # parallelism gives the best single-session throughput.
    ORT_THREADS    = 4

    OUTPUT_DIR     = Path('/kaggle/working')

    # ── Audio — must match training config exactly ─────────────────────────
    SR             = 32000
    CHUNK_DURATION = 5        # seconds per inference chunk
    N_FFT          = 4096
    HOP_LENGTH     = 1252
    N_MELS         = 224
    FMIN           = 0
    FMAX           = 16000

    # ── Model — must match training config exactly ─────────────────────────
    MODEL_NAME     = 'tf_efficientnetv2_s.in21k_ft_in1k'
    DROP_PATH_RATE = 0.15
    USE_GEM        = False    # set True if trained with --use_gem true
    GEM_P_INIT     = 3.0     # must match gem_p_init used during training
    IMAGENET_NORM  = False    # set True for ViT models trained with imagenet_norm=True

    # ── Temperature scaling ────────────────────────────────────────────────
    TAXONOMY_CSV       = BASE_DIR / 'taxonomy.csv'
    TAXON_TEMPERATURES = {'Aves': 1.0, 'Insecta': 1.0, 'Amphibia': 1.0}

    # ── Post-processing ────────────────────────────────────────────────────
    # Temporal TTA: re-run inference with the full waveform circularly rolled
    # by ±TTA_SHIFT samples, then average with the unshifted predictions.
    # Weight: (1 * orig + 2 * roll_fwd + 2 * roll_bwd) / 5
    # ±40,000 samples = ±1.25s at SR=32000. Zero training cost.
    TEMPORAL_TTA       = False
    TTA_SHIFT          = 40_000   # samples

    # Temporal smoothing: [0.20, 0.60, 0.20] kernel. Disabled — 0 to -0.001 LB.
    TEMPORAL_SMOOTHING = False

    # Delta-shift smoothing: gentle temporal smoothing across consecutive chunks.
    # new[t] = (1 - alpha) * old[t] + 0.5 * alpha * (old[t-1] + old[t+1])
    # OOF isolation tests: +0.003 to +0.007 AUC.
    DELTA_SMOOTH       = False
    DELTA_SMOOTH_ALPHA = 0.15

    # Rank-aware scaling: multiply each chunk's predictions by file_max^power
    # per species. Suppresses species the model is never confident about in
    # a given file. OOF isolation tests: +0.002 to +0.017 AUC.
    RANK_AWARE         = False
    RANK_AWARE_POWER   = 0.4

    # Confidence scaling: multiply each chunk by the per-species top-K mean
    # across all chunks of the file. Validated: +0.018–0.027 LB.
    CONF_SCALE         = True
    CONF_SCALE_TOP_K   = 2

    # Diel prior: class-level likelihood ratios from training soundscapes.
    # Disabled — causes -0.006 LB (double-counts soundscape training signal).
    DIEL_PRIOR         = False

    # ── Inference ─────────────────────────────────────────────────────────
    DEVICE     = 'cpu'    # competition constraint
    BATCH_SIZE = 12       # all chunks of one 1-min file in one pass

    # ── I/O prefetch ──────────────────────────────────────────────────────
    # Number of files to load ahead of the main thread in a background thread.
    # torchaudio.load + OGG decode releases the GIL, so this overlaps disk
    # I/O and audio decoding with ONNX inference at no extra CPU contention.
    # 2 is enough to hide latency; more wastes RAM (each file ~7 MB as float32).
    PREFETCH   = 2

    # ── 20s window inference ───────────────────────────────────────────────
    # Set TRAIN_DURATION = 20 to infer with sliding 20s windows (5s stride),
    # matching the window size the model was trained on.  Audio is symmetrically
    # padded by 7.5s on each side so each 5s target chunk is centred in its
    # 20s window.  frame_logits (T_train, C) from each window are placed in a
    # shared timeline and averaged where windows overlap, then max-pooled per
    # 5s chunk (1st-place overlap-average-max reconstruction).
    # Requires swa_model_20s.onnx (export: python export_onnx.py --duration 20).
    # Set TRAIN_DURATION = 5 (default) to keep the original per-chunk behaviour.
    TRAIN_DURATION = 20

    # ── Derived ────────────────────────────────────────────────────────────
    @classmethod
    def chunk_frames(cls):
        """Number of mel time frames for one CHUNK_DURATION clip."""
        return math.floor(cls.SR * cls.CHUNK_DURATION / cls.HOP_LENGTH) + 1

    @classmethod
    def train_frames(cls):
        """Number of mel time frames for one TRAIN_DURATION window."""
        return math.floor(cls.SR * cls.TRAIN_DURATION / cls.HOP_LENGTH) + 1

CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Auto-select the ONNX model file based on TRAIN_DURATION:
#   5s  → swa_model.onnx      (exported with default export_onnx.py)
#   20s → swa_model_20s.onnx  (exported with: python export_onnx.py --duration 20)
# Override CFG.ONNX_MODEL before this line to use a custom path, or set it to
# None to force the PyTorch backend regardless of TRAIN_DURATION.
if CFG.ONNX_MODEL is None:
    _onnx_stem = ('swa_model.onnx' if CFG.TRAIN_DURATION == CFG.CHUNK_DURATION
                  else f'best_sc_model_{CFG.TRAIN_DURATION}s.onnx')
    CFG.ONNX_MODEL = CFG.ARTEFACT_DIR / _onnx_stem

_win_mode = f'{CFG.TRAIN_DURATION}s window' if CFG.TRAIN_DURATION > CFG.CHUNK_DURATION else '5s chunk'
print(f'Chunk frames : {CFG.chunk_frames()}  (mel T for {CFG.CHUNK_DURATION}s at hop={CFG.HOP_LENGTH})')
print(f'Train frames : {CFG.train_frames()}  (mel T for {CFG.TRAIN_DURATION}s — inference mode: {_win_mode})')
print(f'ONNX model   : {CFG.ONNX_MODEL}')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Load Submission Template & Label Map

# %% [code] {"jupyter":{"outputs_hidden":false}}
sample_sub      = pd.read_csv(CFG.SAMPLE_SUB)
SPECIES_COLS    = [c for c in sample_sub.columns if c != 'row_id']
NUM_SUB_SPECIES = len(SPECIES_COLS)
print(f'Submission species : {NUM_SUB_SPECIES}')
print(f'Example row_id     : {sample_sub["row_id"].iloc[0]}')
sample_sub.head(3)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# label_map.npy is saved as a dict {int -> species_str} via np.save.
# np.load returns a 0-d object array; .item() extracts the dict.
label_map   = np.load(CFG.LABEL_MAP, allow_pickle=True).item()
NUM_CLASSES = len(label_map)
idx2species = label_map    # {int: str}

UNIFORM_PRIOR = 1.0 / NUM_SUB_SPECIES

# model output index -> submission column index (-1 = not in submission)
model_to_sub = np.full(NUM_CLASSES, -1, dtype=np.int32)
sub_col_map  = {sp: j for j, sp in enumerate(SPECIES_COLS)}
for model_idx, sp in idx2species.items():
    if sp in sub_col_map:
        model_to_sub[model_idx] = sub_col_map[sp]

n_mapped = (model_to_sub >= 0).sum()
print(f'Model classes                        : {NUM_CLASSES}')
print(f'Mapped to submission columns         : {n_mapped}')
print(f'Submission columns on uniform prior  : {NUM_SUB_SPECIES - n_mapped}')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Load Model
# #
# Tries to load an ONNX model via OnnxRuntime (faster on CPU).
# Falls back to the inline PyTorch BirdCLEFModel if the ONNX file is absent
# or onnxruntime is not installed.
# #
# Inline PyTorch class matches `Codebase/birdclef/model.py::BirdCLEFModel`
# and supports CNN + ViT backbones and optional GeM pooling.
# Must stay in sync with the codebase version.

# %% [code] {"jupyter":{"outputs_hidden":false}}
_VIT_PREFIXES = ('vit_', 'deit_', 'beit_', 'eva_')


def _is_vit(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _VIT_PREFIXES)


class GeMPool1d(nn.Module):
    """GeM pooling over dim=2 of (B, C, H, W). p loaded from checkpoint."""
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class BirdCLEFModel(nn.Module):
    """
    SED architecture matching Codebase/birdclef/model.py::BirdCLEFModel.
    Only used as PyTorch fallback when ONNX is unavailable.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        n_mels: int,
        chunk_frames: int,
        drop_path_rate: float = 0.0,
        use_gem: bool = False,
        gem_p_init: float = 3.0,
    ):
        super().__init__()
        self._is_vit  = _is_vit(model_name)
        self.num_classes = num_classes
        self._use_gem = use_gem
        if use_gem:
            self.gem_pool = GeMPool1d(p_init=gem_p_init)

        if self._is_vit:
            self.backbone = timm.create_model(
                model_name, pretrained=False, num_classes=0, global_pool='',
                in_chans=3, img_size=(n_mels, chunk_frames),
                drop_path_rate=drop_path_rate, dynamic_img_size=True,
            )
            with torch.no_grad():
                dummy      = torch.zeros(1, 3, n_mels, chunk_frames)
                tokens     = self.backbone(dummy)[:, 1:, :]
                n_features = tokens.shape[2]
                self.h_patches = n_mels // self.backbone.patch_embed.patch_size[0]
        else:
            self.backbone = timm.create_model(
                model_name, pretrained=False, num_classes=0, global_pool='',
                in_chans=3, drop_path_rate=drop_path_rate,
            )
            with torch.no_grad():
                dummy      = torch.zeros(1, 3, n_mels, chunk_frames)
                n_features = self.backbone(dummy).shape[1]

        self.fc      = nn.Linear(n_features, num_classes)
        self.att_fc  = nn.Linear(n_features, num_classes)
        self.bn      = nn.BatchNorm1d(n_features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor):
        if self._is_vit:
            tokens = self.backbone(x)[:, 1:, :]              # (B, N, D)
            B, N, D = tokens.shape
            w_patches = N // self.h_patches
            feat = tokens.reshape(B, self.h_patches, w_patches, D).permute(0, 3, 1, 2)
            feat = self.gem_pool(feat) if self._use_gem else feat.mean(dim=2)
            feat = feat.permute(0, 2, 1)                     # (B, T, D)
        else:
            feat = self.backbone(x)                          # (B, C, H, W)
            feat = self.gem_pool(feat) if self._use_gem else feat.mean(dim=2)
            feat = feat.permute(0, 2, 1)                     # (B, T, C)

        B, T, C = feat.shape
        feat            = self.bn(feat.reshape(B * T, C)).reshape(B, T, C)
        feat            = self.dropout(feat)
        frame_logits    = self.fc(feat)
        att_weights     = torch.softmax(self.att_fc(feat), dim=1)
        clipwise_logits = frame_logits.mean(dim=1)
        att_clipwise    = (frame_logits * att_weights).sum(dim=1)
        return clipwise_logits, att_clipwise, frame_logits


# ── Backend selection: ONNX preferred, PyTorch fallback ───────────────────────
# For 20s window inference (TRAIN_DURATION=20) the ONNX model must be exported
# with --duration 20, which produces swa_model_20s.onnx.  That model bakes the
# first-T_chunk frame max-pool into the graph, so its output shape (N, C) is
# identical to the standard 5s model — the same ORT inference path works for
# both modes.  CFG.ONNX_MODEL is auto-set above; re-export if the file is absent.
_use_onnx       = False
_ort_session    = None
_ort_output_dims = 0   # 2 for 5s model (N,C), 3 for 20s model (N,T,C)
model           = None

_onnx_path = CFG.ONNX_MODEL
if _ORT_AVAILABLE and _onnx_path is not None and Path(_onnx_path).exists():
    _sess_opts = ort.SessionOptions()
    _sess_opts.intra_op_num_threads = CFG.ORT_THREADS
    _sess_opts.execution_mode       = ort.ExecutionMode.ORT_SEQUENTIAL
    _ort_session = ort.InferenceSession(
        str(_onnx_path),
        sess_options=_sess_opts,
        providers=['CPUExecutionProvider'],
    )
    _use_onnx        = True
    _ort_output_dims = len(_ort_session.get_outputs()[0].shape)
    print(f'Backend       : OnnxRuntime  ({_onnx_path.name})')
    print(f'ORT threads   : {CFG.ORT_THREADS}')
    print(f'ORT output    : {_ort_session.get_outputs()[0].name}  '
          f'({"(N,C) 5s" if _ort_output_dims == 2 else "(N,T,C) 20s — overlap reconstruction"})')
else:
    if not _ORT_AVAILABLE:
        print('Backend       : PyTorch  (onnxruntime not installed)')
    elif _onnx_path is None:
        print('Backend       : PyTorch  (CFG.ONNX_MODEL is None)')
    else:
        print(f'Backend       : PyTorch  ({Path(_onnx_path).name} not found — '
              f'run: python export_onnx.py --exp_dir <dir>'
              + (f' --duration {CFG.TRAIN_DURATION}' if CFG.TRAIN_DURATION != CFG.CHUNK_DURATION else '')
              + ')')

if not _use_onnx:
    # In 20s mode, initialise the backbone with train_frames so the dummy
    # forward pass sees the correct input width (important for ViT img_size).
    model = BirdCLEFModel(
        model_name     = CFG.MODEL_NAME,
        num_classes    = NUM_CLASSES,
        n_mels         = CFG.N_MELS,
        chunk_frames   = CFG.train_frames(),
        drop_path_rate = CFG.DROP_PATH_RATE,
        use_gem        = CFG.USE_GEM,
        gem_p_init     = CFG.GEM_P_INIT,
    )
    model.load_state_dict(torch.load(CFG.CHECKPOINT, map_location='cpu'))
    model.eval()

print(f'Model classes : {NUM_CLASSES}')
print(f'GeM pooling   : {CFG.USE_GEM}')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4b. Temperature Scaling Setup
# #
# Builds a per-species temperature vector in model-output order.
# Applied as `sigmoid(logits / temp)` in `_forward_probs`.
# All values currently 1.0 (no scaling). Edit CFG.TAXON_TEMPERATURES to tune.

# %% [code] {"jupyter":{"outputs_hidden":false}}
taxonomy  = pd.read_csv(CFG.TAXONOMY_CSV)
taxon_map = taxonomy.set_index('primary_label')['class_name'].to_dict()
taxon_map = {str(k): v for k, v in taxon_map.items()}

model_temp = torch.ones(NUM_CLASSES)
for model_idx, sp in idx2species.items():
    cls = taxon_map.get(str(sp), 'Aves')
    model_temp[model_idx] = CFG.TAXON_TEMPERATURES.get(cls, 1.0)

temp_counts = Counter(model_temp.tolist())
print('Temperature vector built:')
for t, n in sorted(temp_counts.items()):
    print(f'  T={t:.2f}  →  {n} species')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4c. Diel Prior Setup
# #
# Class-level likelihood ratios LR(class | hour) derived from 739 labelled
# 5-second chunks across 66 training soundscape files.
# Disabled by default — causes -0.006 LB (double-counts soundscape training signal).

# %% [code] {"jupyter":{"outputs_hidden":false}}
_DIEL_LR = {
    'Amphibia': {0: 0.776, 1: 1.118, 2: 1.274, 3: 0.333, 4: 0.441,
                 6: 0.100, 7: 0.100, 18: 1.256, 19: 1.289,
                 20: 1.299, 21: 1.300, 22: 1.297, 23: 1.300},
    'Aves':     {0: 1.281, 1: 1.440, 2: 0.942, 3: 0.746, 4: 1.741,
                 6: 3.447, 7: 1.647, 18: 3.348, 19: 0.141,
                 20: 0.102, 21: 0.100, 22: 0.100, 23: 1.549},
    'Insecta':  {0: 0.100, 1: 0.100, 2: 0.100, 3: 3.271, 4: 2.908,
                 6: 4.347, 7: 4.332, 18: 0.169, 19: 2.908,
                 20: 0.100, 21: 0.100, 22: 0.100, 23: 0.100},
}
_HOUR_RE = _re.compile(r'_(\d{8})_(\d{6})')

_diel_log_lr_by_hour = {}
for _h in range(24):
    _log_lr = np.zeros(NUM_CLASSES, dtype=np.float32)
    for _model_idx, _sp in idx2species.items():
        _cls = taxon_map.get(str(_sp))
        if _cls in _DIEL_LR:
            _lr = _DIEL_LR[_cls].get(_h, 1.0)
            _log_lr[_model_idx] = np.log(_lr)
    _diel_log_lr_by_hour[_h] = _log_lr


def apply_diel_prior(probs: np.ndarray, filename: str) -> np.ndarray:
    """
    Adjust (N_chunks, NUM_CLASSES) probabilities by class-level diel LR.
    Extracts hour from filename, applies log-LR shift in logit space.
    Returns probs unchanged if hour cannot be parsed.
    """
    m = _HOUR_RE.search(filename)
    if m is None:
        return probs
    hour   = int(m.group(2)[:2])
    log_lr = _diel_log_lr_by_hour.get(hour, _diel_log_lr_by_hour[0] * 0)
    if not np.any(log_lr):
        return probs
    eps    = 1e-6
    logits = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))
    return (1.0 / (1.0 + np.exp(-(logits + log_lr[np.newaxis, :])))).astype(np.float32)

_test_hour  = 6
_test_probs = np.full((1, NUM_CLASSES), 0.3, dtype=np.float32)
_adjusted   = apply_diel_prior(_test_probs, f'BC2026_Test_0001_S05_20250101_{_test_hour:02d}0000.ogg')
print(f'Diel prior sanity check at hour {_test_hour}:')
for _sp, _cls in [('rufhor2', 'Aves'), ('22985', 'Amphibia'), ('1161364', 'Insecta')]:
    _idx = next((i for i, s in idx2species.items() if str(s) == _sp), None)
    if _idx is not None:
        print(f'  {_sp} ({_cls}): {_test_probs[0, _idx]:.3f} → {_adjusted[0, _idx]:.3f}  '
              f'(LR={_DIEL_LR.get(_cls, {}).get(_test_hour, 1.0):.3f})')
print('Diel prior setup complete.')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Audio Utilities

# %% [code] {"jupyter":{"outputs_hidden":false}}
mel_transform   = T.MelSpectrogram(
    sample_rate = CFG.SR,
    n_fft       = CFG.N_FFT,
    hop_length  = CFG.HOP_LENGTH,
    n_mels      = CFG.N_MELS,
    f_min       = CFG.FMIN,
    f_max       = CFG.FMAX,
)
amplitude_to_db = T.AmplitudeToDB(top_db=80)

# ImageNet normalisation constants — only applied when CFG.IMAGENET_NORM is True
# (ViT models trained with imagenet_norm=True in the training config).
_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IN_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_and_chunk(filepath: Path) -> tuple[torch.Tensor, int]:
    """
    Load a soundscape, resample if needed, and return audio windows.

    5s mode  (TRAIN_DURATION == CHUNK_DURATION):
        Non-overlapping 5s chunks, same as before.
        Returns (N_chunks, chunk_len).

    20s mode (TRAIN_DURATION > CHUNK_DURATION):
        Overlapping TRAIN_DURATION-second windows with CHUNK_DURATION stride.
        Window i covers [i*step_len, i*step_len + win_len].
        The waveform is zero-padded so the last window is always full.
        Returns (N_chunks, win_len).

    In both cases N_chunks = floor(file_duration / CHUNK_DURATION).
    """
    waveform, orig_sr = torchaudio.load(str(filepath))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != CFG.SR:
        waveform = torchaudio.functional.resample(waveform, orig_sr, CFG.SR)

    chunk_len = CFG.SR * CFG.CHUNK_DURATION
    flat      = waveform.squeeze(0)

    # Number of complete 5s chunks in the file
    n_chunks  = flat.shape[0] // chunk_len

    if CFG.TRAIN_DURATION == CFG.CHUNK_DURATION:
        # Original path: non-overlapping 5s chunks.
        # Pad to the next multiple of chunk_len, then truncate to n_chunks rows.
        # Truncation is needed because files that are a few samples over a
        # chunk boundary (e.g. 60.001s) would otherwise yield n_chunks+1 rows
        # after reshape while n_chunks was pre-computed without the pad tail.
        remainder = flat.shape[0] % chunk_len
        if remainder:
            flat = F.pad(flat, (0, chunk_len - remainder))
        return flat.reshape(-1, chunk_len)[:n_chunks], n_chunks

    # 20s window path: overlapping windows with CHUNK_DURATION stride.
    # Symmetric audio padding by half-overlap ensures the target 5s chunk is
    # centred in each 20s window.  Without this, _overlap_average_max trims
    # (T_train - T_chunk)//2 ≈ 15 backbone-frames = 7.5s from the front,
    # shifting every chunk's prediction forward by 7.5s.
    win_len  = CFG.SR * CFG.TRAIN_DURATION
    step_len = chunk_len
    sym_pad  = (win_len - step_len) // 2   # = 7.5 s at SR=32000 / 20s window
    needed   = sym_pad + (n_chunks - 1) * step_len + win_len
    flat = F.pad(flat, (sym_pad, max(0, needed - sym_pad - flat.shape[0])))
    windows = torch.stack([flat[i * step_len : i * step_len + win_len]
                           for i in range(n_chunks)])
    return windows, n_chunks


def chunks_to_mel_batch(chunks: torch.Tensor) -> torch.Tensor:
    """
    Convert (N_chunks, chunk_len) → (N_chunks, 3, N_MELS, T_frames).
    Vectorised; no Python loop per chunk.
    Applies ImageNet normalisation when CFG.IMAGENET_NORM is True.
    """
    mel = mel_transform(chunks)                           # (N, n_mels, T)
    mel = amplitude_to_db(mel)
    mn  = mel.flatten(1).min(dim=1).values[:, None, None]
    mx  = mel.flatten(1).max(dim=1).values[:, None, None]
    mel = (mel - mn) / (mx - mn + 1e-6)
    mel = mel.unsqueeze(1).repeat(1, 3, 1, 1)            # (N, 3, n_mels, T)
    if CFG.IMAGENET_NORM:
        mel = (mel - _IN_MEAN) / _IN_STD
    return mel

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 6. Inference Loop

# %% [code] {"jupyter":{"outputs_hidden":false}}
_model_temp_np = None   # lazily set to model_temp.numpy() on first ORT call


def _overlap_average_max(
    frame_probs: np.ndarray,   # (n_windows, T_train, C) probabilities in [0, 1]
    T_train: int,
    T_chunk: int,
) -> np.ndarray:
    """
    1st-place overlap-average-max reconstruction.

    Each of the n_windows 20s predictions is placed in a shared per-frame
    timeline at position [i*T_chunk, i*T_chunk + T_train).  Positions covered
    by multiple windows are averaged — this cancels contamination from adjacent
    chunks that leaks through the backbone's receptive field.

    The timeline is then trimmed symmetrically by pad = (T_train-T_chunk)//2
    frames from each end (where fewer windows overlap and edge effects dominate),
    leaving exactly n_windows * T_chunk frames.  A final max-pool within each
    chunk's T_chunk frames produces (n_windows, C) probabilities.

    Returns (n_windows, C) float32 probabilities.
    """
    n_chunks = frame_probs.shape[0]
    C        = frame_probs.shape[2]
    step     = T_chunk
    ss_len   = T_train + step * (n_chunks - 1)

    timeline = np.zeros((ss_len, C), dtype=np.float32)
    count    = np.zeros((ss_len, 1), dtype=np.float32)

    for i in range(n_chunks):
        s = i * step
        timeline[s : s + T_train] += frame_probs[i]    # (T_train, C)
        count   [s : s + T_train] += 1.0

    timeline /= count

    # Trim symmetric padding so the remaining length is exactly n_chunks * step
    pad   = (T_train - step) // 2
    extra = (T_train - step) % 2
    timeline = timeline[pad : ss_len - pad - extra]     # (n_chunks * step, C)

    return timeline.reshape(n_chunks, step, C).max(axis=1)   # (n_chunks, C)


def _forward_probs(batch: torch.Tensor) -> np.ndarray:
    """
    Forward pass → probabilities (N_chunks, NUM_CLASSES).

    5s mode (TRAIN_DURATION == CHUNK_DURATION):
        att_clipwise from the 5s window — unchanged from the original script.

    20s mode (TRAIN_DURATION > CHUNK_DURATION):
        frame_logits (N, T_train, C) are obtained from either ONNX or PyTorch,
        then fed into _overlap_average_max which replicates the 1st-place
        overlap-average-max reconstruction:
          1. Sigmoid + temperature scaling per frame
          2. Place all 12 window predictions in a shared per-frame timeline
          3. Average overlapping regions (cancels cross-chunk contamination)
          4. Trim symmetric edge padding
          5. Max-pool within each 5s chunk → (N, C) probabilities

    Temperature scaling is applied before sigmoid in both modes.
    """
    if _use_onnx:
        global _model_temp_np
        if _model_temp_np is None:
            _model_temp_np = model_temp.numpy()

        raw = _ort_session.run(None, {'mel_spec': batch.numpy()})[0]

        if _ort_output_dims == 2:
            # 5s model: (N, C) logits → sigmoid
            return (1.0 / (1.0 + np.exp(-raw / _model_temp_np))).astype(np.float32)

        # 20s model: (N, T_train, C) logits → per-frame sigmoid → reconstruction
        T_train     = raw.shape[1]
        T_chunk     = max(1, T_train * CFG.CHUNK_DURATION // CFG.TRAIN_DURATION)
        frame_probs = (1.0 / (1.0 + np.exp(-raw / _model_temp_np))).astype(np.float32)
        return _overlap_average_max(frame_probs, T_train, T_chunk)

    with torch.inference_mode():
        _, att_clipwise, frame_logits = model(batch)

        if CFG.TRAIN_DURATION == CFG.CHUNK_DURATION:
            return torch.sigmoid(att_clipwise / model_temp).numpy()

        # 20s mode: per-frame sigmoid then overlap reconstruction
        T_train     = frame_logits.shape[1]
        T_chunk     = max(1, T_train * CFG.CHUNK_DURATION // CFG.TRAIN_DURATION)
        frame_probs = torch.sigmoid(frame_logits / model_temp).numpy()  # (N, T_train, C)
        return _overlap_average_max(frame_probs, T_train, T_chunk)


def _roll_chunks(chunks: torch.Tensor, shift: int) -> torch.Tensor:
    """Circularly roll the full waveform by `shift` samples, then re-chunk."""
    flat     = chunks.reshape(-1)
    flat     = torch.roll(flat, shifts=shift)
    n_chunks = chunks.shape[0]
    return flat.reshape(n_chunks, -1)


def _process_chunks(filepath: Path, chunks: torch.Tensor, n_chunks: int) -> pd.DataFrame:
    """
    Compute-only part of inference: mel → ONNX → post-processing → DataFrame.
    Separated from I/O so the main loop can overlap loading with inference.

    Post-processing order:
      1. Temporal TTA        — average over ±1.25s circular rolls (1:2:2 weights)
      2. Diel prior          — class-level LR in logit space (disabled: -0.006 LB)
      3. Temporal smoothing  — [0.20, 0.60, 0.20] kernel (disabled: 0 to -0.001 LB)
      4. Delta-shift smooth  — gentle [0.075, 0.85, 0.075] kernel (+0.003 to +0.007)
      5. Rank-aware scaling  — multiply by file_max^power per species (+0.002 to +0.017)
      6. Confidence scaling  — multiply by per-species top-K mean (+0.018–0.027 LB)
    """
    stem  = filepath.stem

    probs = _forward_probs(chunks_to_mel_batch(chunks))   # (N, NUM_CLASSES)

    if CFG.TEMPORAL_TTA:
        probs_fwd = _forward_probs(chunks_to_mel_batch(_roll_chunks(chunks, +CFG.TTA_SHIFT)))
        probs_bwd = _forward_probs(chunks_to_mel_batch(_roll_chunks(chunks, -CFG.TTA_SHIFT)))
        probs = (probs + 2.0 * probs_fwd + 2.0 * probs_bwd) / 5.0

    if CFG.DIEL_PRIOR:
        probs = apply_diel_prior(probs, filepath.name)

    if CFG.TEMPORAL_SMOOTHING:
        padded = np.concatenate([probs[:1], probs, probs[-1:]], axis=0)
        probs  = 0.20 * padded[:-2] + 0.60 * padded[1:-1] + 0.20 * padded[2:]

    if CFG.DELTA_SMOOTH:
        a     = CFG.DELTA_SMOOTH_ALPHA
        prev  = np.concatenate([probs[:1],  probs[:-1]], axis=0)
        nxt   = np.concatenate([probs[1:],  probs[-1:]], axis=0)
        probs = (1 - a) * probs + 0.5 * a * (prev + nxt)

    if CFG.RANK_AWARE:
        file_max = probs.max(axis=0)
        scale    = np.power(np.clip(file_max, 1e-9, 1.0), CFG.RANK_AWARE_POWER)
        probs    = probs * scale[None, :]

    if CFG.CONF_SCALE:
        top_k_mean = np.sort(probs, axis=0)[-CFG.CONF_SCALE_TOP_K:].mean(axis=0)
        probs      = probs * top_k_mean[None, :]

    _exp85_mapped = model_to_sub >= 0
    out = np.full((n_chunks, NUM_SUB_SPECIES), UNIFORM_PRIOR, dtype=np.float32)
    out[:, model_to_sub[_exp85_mapped]] = probs[:, _exp85_mapped]

    row_ids = [f'{stem}_{(i + 1) * CFG.CHUNK_DURATION}' for i in range(n_chunks)]
    df = pd.DataFrame(out, columns=SPECIES_COLS)
    df.insert(0, 'row_id', row_ids)
    return df


# ── Main inference loop ────────────────────────────────────────────────────
test_files = sorted(CFG.TEST_DIR.glob('*.ogg'))
print(f'Test files found: {len(test_files)}')

if len(test_files) == 0:
    print('Hidden test not mounted. Dry-run on first 20 train soundscapes.')
    test_files = sorted(CFG.SOUNDSCAPE_DIR.glob('*.ogg'))[:n_dry_run]

all_preds = []
t0        = time.time()

# Prefetch pipeline: a thread pool keeps PREFETCH load_and_chunk futures in
# flight while the main thread runs mel + ONNX + post-processing.
# torchaudio.load and OGG decoding release the GIL, so this overlaps disk
# I/O with compute without competing with ORT's CPU threads.
with ThreadPoolExecutor(max_workers=CFG.PREFETCH) as pool:
    # Dict-based prefetch: pop each future after use so its cached 20s audio
    # tensor (~30 MB per file) is freed immediately instead of accumulating
    # to 18+ GB across 600 files (Python Future caches result in self._result).
    pending  = {}
    files    = list(test_files)
    n_files  = len(files)
    next_idx = 0

    # Seed: submit first PREFETCH files
    for i, f in enumerate(files[:CFG.PREFETCH]):
        pending[i] = pool.submit(load_and_chunk, f)
    next_idx = CFG.PREFETCH

    for i, filepath in enumerate(tqdm(files, total=n_files, desc='Inference')):
        # Submit the next file immediately so it loads while we do inference
        if next_idx < n_files:
            pending[next_idx] = pool.submit(load_and_chunk, files[next_idx])
            next_idx += 1

        # pop() drops the Future reference; the cached audio tensor is freed
        # when `chunks` goes out of scope at the end of this iteration.
        chunks, n_chunks = pending.pop(i).result()
        all_preds.append(_process_chunks(filepath, chunks, n_chunks))

elapsed  = time.time() - t0
n_passes = 3 if CFG.TEMPORAL_TTA else 1
print(f'\nDone in {elapsed / 60:.1f} min  ({elapsed / len(test_files):.2f} s/file, {n_passes} pass(es)/file)')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 7. Build & Validate Submission

# %% [code] {"jupyter":{"outputs_hidden":false}}
preds = pd.concat(all_preds, ignore_index=True)

submission = preds
for sp in SPECIES_COLS:
    submission[sp] = submission[sp].fillna(UNIFORM_PRIOR)

print(f'Submission shape  : {submission.shape}')
print(f'Rows              : {submission.shape[0]}')
print(f'Species columns   : {submission.shape[1] - 1}')
print(f'All checks passed ✓')
submission.head(3)

# %% [code] {"jupyter":{"outputs_hidden":false}}
_PROBS_CACHE["exp85"] = (submission["row_id"].values.copy(), submission[SPECIES_COLS].to_numpy(np.float32))
print(f'exp85 probs cached  shape={submission.shape}')

import gc

# Keep `preds` (raw per-model predictions) for ensemble combination.
# Delete the model session, audio pipeline, and intermediate buffers.

if _use_onnx and _ort_session is not None:
    del _ort_session
elif model is not None:
    del model
del mel_transform, amplitude_to_db
del all_preds, preds, submission
del model_temp, _model_temp_np, _diel_log_lr_by_hour
gc.collect()
print('exp85_preds ready for ensembling.')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Ensembling

# %% [code] {"jupyter":{"outputs_hidden":false}}
import re as _re
import numpy as np
import pandas as pd
from pathlib import Path

# ── OUTPUTS ───────────────────────────────────────────────────────────────
OUTPUT_CSV = "submission.csv"
EPS        = 1e-5

# ── MODEL REGISTRY ────────────────────────────────────────────────────────
# To add a model: append an entry with its csv filename (relative to
# CFG.OUTPUT_DIR), is_anchor=False, and per-taxon blend weights.
# Only one model should have is_anchor=True — this is the model used for
# temporal smoothing (Gate 2) and noise/spike detection (Gates 1 & 3).
# Weight ratios matter; values are normalised per species at blend time.

MODELS = [
    {
        "name":      "perch",
        "csv":       "submission_perch.csv",
        "is_anchor": True,
        "w": {"Aves": 0.25, "Amphibia": 0.25, "Insecta": 0.25,
              "Mammalia": 0.25, "Reptilia": 0.25},
    },
    {
        "name":      "sed",
        "csv":       "submission_sed.csv",
        "is_anchor": False,
        "w": {"Aves": 0.25, "Amphibia": 0.25, "Insecta": 0.25,
              "Mammalia": 0.25, "Reptilia": 0.25},
    },
    {
        "name":      "exp53",
        "csv":       "submission_exp53.csv",
        "is_anchor": False,
        "w": {"Aves": 0.25, "Amphibia": 0.25, "Insecta": 0.25,
              "Mammalia": 0.25, "Reptilia": 0.25},
    },    
    {
        "name":      "exp72e",
        "csv":       "submission_exp72e.csv",
        "is_anchor": False,
        "w": {"Aves": 0.20, "Amphibia": 0.20, "Insecta": 0.17,
              "Mammalia": 0.20, "Reptilia": 0.20},
    },
    {
        "name":      "exp79b",
        "csv":       "submission_exp79b.csv",
        "is_anchor": False,
        "w": {"Aves": 0.20, "Amphibia": 0.20, "Insecta": 0.17,
              "Mammalia": 0.20, "Reptilia": 0.20},
    },
    {
        "name":      "exp85",
        "csv":       "submission_exp85.csv",
        "is_anchor": False,
        "w": {"Aves": 0.0, "Amphibia": 0.0, "Insecta": 1.25,
              "Mammalia": 0.0, "Reptilia": 0.0},
    },
    # Uncomment to test exp88 (also uncomment in _EFFNET_MODELS above):
     {
         "name":      "exp88",
         "csv":       "submission_exp88.csv",
         "is_anchor": False,
         "w": {"Aves": 0.20, "Amphibia": 0.20, "Insecta": 0.16,
               "Mammalia": 0.20, "Reptilia": 0.20},
     },
]

# ── GATE TOGGLES ──────────────────────────────────────────────────────────
GATE1_NOISE    = False    # Perch confident + all CNNs disagree  → trust Perch
GATE2_TEMPORAL = True    # Fat-tailed temporal continuity on Perch probs
GATE3_SPIKE    = False    # Any CNN spike Perch missed           → soft boost
GATE4_MIRROR   = True    # Sonotype max-pooling  (keep on)
GATE5_RARE     = False    # Rare-class suppression (Amphibia / Mammalia / Reptilia)
GATE6_STATION  = False    # Station-level floor for guaranteed-present sonotypes
GATE_SR        = False    # SoftmaxRichness per-chunk normalisation (applied last)

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────
G1_P_ANCHOR    = 0.50   # Perch raw-prob threshold for "confident"
G1_P_CNN       = 0.05   # CNN mean raw-prob threshold for "all disagree"
G1_ALPHA       = 0.08   # Fraction of r_anchor blended in

G2_CTX_RANK    = 0.88   # Smoothed-context rank threshold
G2_ANCHOR_RANK = 0.75   # Perch rank threshold
G2_ALPHA       = 0.15   # Fraction of max(r_anchor, xctx) blended in

G3_CNN_RANK    = 0.95   # Any-CNN rank threshold for "spike"
G3_ANCHOR_RANK = 0.80   # Perch rank threshold for "missed"
G3_ALPHA       = 0.12   # Fraction of r_cnn_max blended in

STATION_FLOORS     = {"S15": ["47158son07"], "S23": ["47158son25"]}
STATION_FLOOR_RANK = 0.65   # minimum blended-rank for guaranteed-present species

SR_T = 0.15             # SoftmaxRichness temperature (lower = sharper attention)

# ── MIRROR GROUPS ─────────────────────────────────────────────────────────
# Acoustically identical sonotype groups — max-pooled so detecting any one
# member boosts the whole group.
MIRROR_PAIRS = (
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23"),
)

# ═══════════════════════════════════════════════════════════════════════════
# Load & align — read from in-memory cache; fall back to CSV if not cached
# ═══════════════════════════════════════════════════════════════════════════
_ENS_COLS = PRIMARY_LABELS  # same source as every per-model section
for md in MODELS:
    if md["name"] in _PROBS_CACHE:
        _row_ids_m, _p_m = _PROBS_CACHE[md["name"]]
        md["df"] = pd.DataFrame(_p_m, columns=_ENS_COLS)
        md["df"].insert(0, "row_id", _row_ids_m)
    else:
        md["df"] = pd.read_csv(Path("/kaggle/working") / md["csv"])

anchor_df  = next(md["df"] for md in MODELS if md["is_anchor"])
cols       = [c for c in anchor_df.columns if c != "row_id"]
col_to_idx = {label: i for i, label in enumerate(cols)}
row_ids    = anchor_df["row_id"].astype(str).to_numpy()
file_ids   = np.array(["_".join(r.split("_")[:-1]) for r in row_ids])

for md in MODELS:
    md["df"] = md["df"].set_index("row_id").loc[anchor_df["row_id"]].reset_index()
    md["p"]  = np.clip(md["df"][cols].to_numpy(np.float32), EPS, 1 - EPS)
    md["r"]  = pd.DataFrame(md["p"], columns=cols).rank(axis=0, pct=True).to_numpy(np.float32)

anchor   = next(md for md in MODELS if md["is_anchor"])
cnns     = [md for md in MODELS if not md["is_anchor"]]
p_anchor = anchor["p"]
r_anchor = anchor["r"]

# ═══════════════════════════════════════════════════════════════════════════
# Taxon-weighted rank blend
# ═══════════════════════════════════════════════════════════════════════════
taxon_per_col = np.array([CLASS_NAME_MAP.get(label, "Aves") for label in cols])
for md in MODELS:
    md["w_vec"] = np.array([md["w"].get(t, 1.0) for t in taxon_per_col],
                           dtype=np.float32)

total_w = sum(md["w_vec"] for md in MODELS)                         # (n_cols,)
blended = sum(md["w_vec"] * md["r"] for md in MODELS) / total_w    # (n_rows, n_cols)
blended = blended.astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════
# Gates 1-3  (operate on blended rank array; masks accumulate)
# ═══════════════════════════════════════════════════════════════════════════

# ── Gate 1: Noise suppression ──────────────────────────────────────────────
# Perch confident but all CNNs strongly disagree → trust Perch rank more.
fake_only = np.zeros(blended.shape, dtype=bool)
if GATE1_NOISE and cnns:
    p_cnn_mean = np.mean([md["p"] for md in cnns], axis=0)
    fake_only  = (p_anchor > G1_P_ANCHOR) & (p_cnn_mean < G1_P_CNN)
    blended    = np.where(fake_only,
                          (1 - G1_ALPHA) * blended + G1_ALPHA * r_anchor,
                          blended)
    print(f"Gate 1 (noise):    {fake_only.sum():,} cells adjusted.")

# ── Gate 2: Temporal continuity ────────────────────────────────────────────
# Fat-tailed kernel (±3 chunks = ±15 s) on Perch raw probs; boosts species
# whose Perch context rank is high but blended rank may be pulled down by
# a single uncertain chunk.
proto_cont = np.zeros(blended.shape, dtype=bool)
if GATE2_TEMPORAL:
    offs   = np.arange(-3, 4, dtype=np.float32)
    kernel = (1.0 + (offs / 1.20) ** 2 / 2.0) ** (-1.5)
    kernel = (kernel / kernel.sum()).astype(np.float32)
    pa_ctx = p_anchor.copy()
    for fid in pd.unique(file_ids):
        fmask = file_ids == fid
        x     = p_anchor[fmask]
        if len(x) > 1:
            xp            = np.pad(x, ((3, 3), (0, 0)), mode="edge")
            pa_ctx[fmask] = sum(kernel[i] * xp[i: i + len(x)] for i in range(7))
    xctx       = pd.DataFrame(pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)
    proto_cont = (xctx > G2_CTX_RANK) & (r_anchor > G2_ANCHOR_RANK) & (~fake_only)
    blended    = np.where(proto_cont,
                          (1 - G2_ALPHA) * blended
                          + G2_ALPHA * np.maximum(r_anchor, xctx),
                          blended)
    print(f"Gate 2 (temporal): {proto_cont.sum():,} cells adjusted.")

# ── Gate 3: CNN spike preservation ────────────────────────────────────────
# If any CNN ranks a species very high but Perch missed it, add a soft boost.
if GATE3_SPIKE and cnns:
    r_cnn_max = np.max([md["r"] for md in cnns], axis=0)
    cnn_only  = ((r_cnn_max > G3_CNN_RANK) & (r_anchor < G3_ANCHOR_RANK)
                 & (~fake_only) & (~proto_cont))
    blended   = np.where(cnn_only,
                         (1 - G3_ALPHA) * blended + G3_ALPHA * r_cnn_max,
                         blended)
    print(f"Gate 3 (CNN spike): {cnn_only.sum():,} cells adjusted.")

# ── Gate 6: Station-level priors ──────────────────────────────────────────
# Raise blended rank to a floor for sonotypes present in 100 % of training
# chunks at that station.  Only 100 %-reliable priors included.
if GATE6_STATION:
    file_stations = np.array([
        m.group(1) if (m := _re.search(r"_(S\d+)_", fid)) else ""
        for fid in file_ids
    ])
    n_total = 0
    for station, sons in STATION_FLOORS.items():
        row_mask = file_stations == station
        if not row_mask.any():
            continue
        for son in sons:
            if son not in col_to_idx:
                continue
            ci       = col_to_idx[son]
            n_raised = int((blended[row_mask, ci] < STATION_FLOOR_RANK).sum())
            blended[row_mask, ci] = np.maximum(blended[row_mask, ci], STATION_FLOOR_RANK)
            n_total += n_raised
            print(f"  Gate 6: {station} → {son}: {n_raised} rows raised to {STATION_FLOOR_RANK:.2f}")
    print(f"Gate 6 (station):  {n_total} total adjustments.")

# ═══════════════════════════════════════════════════════════════════════════
# DataFrame-level gates (Gate 4, Gate 5) — operate before SoftmaxRichness
# so thresholds stay in the rank-percentile scale.
# ═══════════════════════════════════════════════════════════════════════════
sub       = anchor_df.copy()
sub[cols] = blended

# ── Gate 4: Sonotype mirroring ────────────────────────────────────────────
if GATE4_MIRROR:
    mirror_count = 0
    for group in MIRROR_PAIRS:
        valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
        if len(valid_idx) >= 2:
            group_max = sub[cols].iloc[:, valid_idx].max(axis=1).to_numpy(np.float32)
            for idx in valid_idx:
                sub.iloc[:, idx + 1] = group_max
            mirror_count += len(valid_idx)
    print(f"Gate 4 (mirror):   {mirror_count} columns max-pooled.")

# ── Gate 5: Rare-class suppression ────────────────────────────────────────
if GATE5_RARE:
    try:
        tax_df       = pd.read_csv(CFG.BASE_DIR / "taxonomy.csv").set_index("primary_label")
        rare_classes = {"Amphibia", "Mammalia", "Reptilia"}
        rare_count   = 0
        for ci, species in enumerate(cols):
            if species in tax_df.index and tax_df.loc[species, "class_name"] in rare_classes:
                col_idx = ci + 1
                vals    = sub.iloc[:, col_idx].to_numpy(np.float32)
                thr     = vals.mean() + 0.05
                sub.iloc[:, col_idx] = np.where(vals < thr, vals * 0.9, vals)
                rare_count += 1
        print(f"Gate 5 (rare):     {rare_count} species suppressed.")
    except Exception as e:
        print(f"Gate 5 (rare) skipped: {e}")

# ── SoftmaxRichness ───────────────────────────────────────────────────────
# Applied last so Gates 4 & 5 thresholds operate in consistent rank space.
# Per chunk, weight each species score by its softmax attention (temperature
# SR_T).  Dominant species are preserved; weak species in "rich" noisy chunks
# are suppressed, improving discrimination between real and spurious calls.
if GATE_SR:
    arr  = sub[cols].to_numpy(np.float32)
    z    = arr / SR_T
    z   -= z.max(axis=1, keepdims=True)          # numerical stability
    sm   = np.exp(z)
    sm  /= sm.sum(axis=1, keepdims=True)         # (n_rows, n_cols), sums to 1
    sub[cols] = (arr * sm).astype(np.float32)    # element-wise; stays in [0, 1]
    print(f"SoftmaxRichness:   applied (T={SR_T}).")

# ═══════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════
sub.to_csv(OUTPUT_CSV, index=False)

active = [g for g, on in [("G1-noise", GATE1_NOISE), ("G2-temporal", GATE2_TEMPORAL),
                           ("G3-spike", GATE3_SPIKE),  ("G4-mirror",  GATE4_MIRROR),
                           ("G5-rare",  GATE5_RARE),   ("G6-station", GATE6_STATION),
                           ("SR",       GATE_SR)] if on]
print(f"\nSaved  {OUTPUT_CSV}  shape={sub.shape}")
print(f"Active gates: {active}")
print("\nModel weights (normalised per taxon):")
for taxon in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
    total = sum(md["w"].get(taxon, 1.0) for md in MODELS)
    wstr  = "  ".join(f"{md['name']}={md['w'].get(taxon, 1.0)/total:.2f}" for md in MODELS)
    print(f"  {taxon:10s}: {wstr}")
