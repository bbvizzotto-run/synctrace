"""
SyncTrace — Benchmark Dataloaders
===================================
Loads authentic talking-head clips from the standard audio-visual deepfake
benchmarks and feeds them to the PseudoForgeryGenerator.

Datasets (all open-access, request via their official pages):
  - FakeAVCeleb:      https://zenodo.org/records/7259623  (real videos only)
  - AV-LipSync-TIMIT: https://github.com/omkar137/avspoof  (real videos only)
  - IDForge (optional extension): multimodal forgery benchmark

Only REAL videos are ever loaded into the training loop. Fake samples are
produced on-the-fly by the pseudo-forgery generator with severity labels,
so no synthetic dataset is ever required for training.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class AVClip:
    """An authentic audio-visual clip."""
    video_path: Path
    audio_path: Path
    speaker_id: str
    metadata: dict


class FakeAVCelebLoader:
    """Loads REAL videos from the FakeAVCeleb benchmark.

    Expected layout (after extraction):
      fakeavceleb/
        VideoData/
          Real/
            <language>/
              <speaker_id>/<clip_name>.mp4
    """

    LANGUAGES = ("English", "Hindi", "Tamil", "Telugu")

    def __init__(self, root: Path, sample_rate: int = 16000,
                 max_duration_s: float = 8.0, max_clips: int | None = None):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_duration_s = max_duration_s
        self.max_clips = max_clips

    def _iter_videos(self) -> list[Path]:
        base = self.root / "VideoData" / "Real"
        if not base.exists():
            raise FileNotFoundError(
                f"FakeAVCeleb root not found at {base}. Download from "
                "https://zenodo.org/records/7259623 and extract.")
        videos = []
        for lang in self.LANGUAGES:
            lang_dir = base / lang
            if not lang_dir.exists():
                continue
            videos.extend(sorted(lang_dir.rglob("*.mp4")))
        return videos[: self.max_clips] if self.max_clips else videos

    def load(self) -> list[AVClip]:
        clips = []
        for video in self._iter_videos():
            speaker = video.parent.name
            clips.append(AVClip(
                video_path=video, audio_path=video,  # audio extracted below
                speaker_id=speaker,
                metadata={"benchmark": "fakeavceleb", "language": video.parent.parent.name},
            ))
        return clips


class AVLipSyncTIMITLoader:
    """Loads REAL videos from AV-LipSync-TIMIT.

    Expected layout (after extraction):
      avspoof/
        TIMIT/
          <speaker_id>.mp4
    """

    def __init__(self, root: Path, sample_rate: int = 16000,
                 max_duration_s: float = 8.0, max_clips: int | None = None):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_duration_s = max_duration_s
        self.max_clips = max_clips

    def _iter_videos(self) -> list[Path]:
        base = self.root / "TIMIT"
        if not base.exists():
            raise FileNotFoundError(
                f"AV-LipSync-TIMIT root not found at {base}. Clone from "
                "https://github.com/omkar137/avspoof")
        return sorted(base.glob("*.mp4"))[: self.max_clips]

    def load(self) -> list[AVClip]:
        return [
            AVClip(video_path=v, audio_path=v, speaker_id=v.stem,
                   metadata={"benchmark": "avlipsync-timit"})
            for v in self._iter_videos()
        ]


# ---------------------------------------------------------------------------
# Frame / audio extraction utilities
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, max_frames: int = 32,
                   target_size: int = 112, fps: float = 25.0) -> np.ndarray:
    """Extracts up to max_frames evenly spaced frames at target fps.

    Returns (F, H, W, 3) BGR uint8.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        n_frames = int(cap.get(cv2.CAP_PROP_DURATION_MS) / 1000 * src_fps)
    n_frames = max(1, n_frames)

    # evenly spaced frame selection up to target fps
    indices = np.linspace(0, n_frames - 1, max_frames).astype(int)

    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in set(indices) and len(frames) < max_frames:
            frame = cv2.resize(frame, (target_size, target_size))
            frames.append(frame)
        idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"no frames extracted from {video_path}")
    arr = np.stack(frames)
    # pad or truncate to exactly max_frames
    if arr.shape[0] < max_frames:
        pad = np.repeat(arr[-1:], max_frames - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad])
    return arr[:max_frames]


def extract_audio(video_or_audio_path: Path, sample_rate: int = 16000,
                  max_samples: int = 16000 * 8) -> np.ndarray:
    """Extracts mono waveform via ffmpeg (graceful fallback to zeros)."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_or_audio_path),
            "-ac", "1", "-ar", str(sample_rate),
            "-t", str(max_samples / sample_rate),
            tmp.name,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            from scipy.io import wavfile
            _, data = wavfile.read(tmp.name)
            audio = data.astype(np.float32) / (2 ** 15)
        except (subprocess.CalledProcessError, FileNotFoundError):
            audio = np.zeros(max_samples, dtype=np.float32)
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    return audio[:max_samples]
