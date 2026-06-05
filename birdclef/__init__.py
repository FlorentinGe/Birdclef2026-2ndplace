# BirdCLEF+ 2026 training library
from .config import Config, load_config
from .datasets import (
    BirdDataset,
    BirdDatasetWithPL,
    ExternalNoiseAugmenter,
    FocalValDataset,
    NocallAugmenter,
    PerchDistillDataset,
    SoundscapeTrainDataset,
    SoundscapeValDataset,
)
from .model import BirdCLEFModel, PretrainModel, build_optimizer_with_llrd
from .train import run_kd_stage, run_supervised_stage
from .transforms import MelTransform, mel_to_spectrogram_cpu
from .utils import (
    build_vocabulary,
    build_wav_path_map,
    encode_multilabel,
    make_sc_label_vec,
    normalise_sc_df,
    resolve_audio_paths,
    resolve_wav_paths,
    safe_copy_bn_buffers,
    seed_everything,
    stratified_soundscape_split,
)
from .validate import save_oof_predictions, validate_composite
