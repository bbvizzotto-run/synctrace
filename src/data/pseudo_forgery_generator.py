"""
SyncTrace — Controlled-Severity Pseudo-Forgery Generator
=========================================================
Generates identity-preserving pseudo-forgeries from authentic videos with a
CONTINUOUS severity label (0.0 .. 1.0), enabling:
  (a) self-supervised contrastive training without synthetic deepfake labels
  (b) severity regression (first of its kind in AV deepfake detection)
  (c) automatic ground truth for localization evaluation (per-region masks)

Two manipulation modes:
  1. VISUAL: viseme-level re-synthesis of the lower face (lips/lower-face blend)
  2. AUDIO:  cross-speaker audio swap with controlled temporal offset
"""

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ForgeryRecipe:
    """Describes one pseudo-forgery with controllable severity."""
    visual_severity: float = 0.0   # 0.0 = untouched, 1.0 = full region replaced
    audio_severity: float = 0.0    # 0.0 = original audio, 1.0 = swapped audio
    audio_offset_ms: int = 0       # extra lip-sync jitter in ms (0..500)
    region: str = "lips"           # "lips" | "lower_face" | "full_face"

    @property
    def combined_severity(self) -> float:
        return min(1.0, self.visual_severity + self.audio_severity)


class PseudoForgeryGenerator:
    """Deterministic (seeded) pseudo-forgery generator with region masks."""

    def __init__(self, seed: int = 42, detection_backend: str = "opencv"):
        self.rng = random.Random(seed)
        self.backend = detection_backend

    # ------------------------------------------------------------------
    # 1. FACE LOCALIZATION (pluggable backend: OpenCV Haar | MediaPipe | YOLO)
    # ------------------------------------------------------------------
    def _detect_face_boxes(self, frame: np.ndarray):
        """Returns list of (x, y, w, h); stub uses OpenCV Haar cascade."""
        if self.backend == "opencv":
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cascade.detectMultiScale(gray, 1.1, 4)
        raise NotImplementedError(f"backend {self.backend} not implemented")

    def _region_mask(self, frame, box, region: str) -> np.ndarray:
        """Binary mask (H, W) for the requested facial region."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x, y, bw, bh = box
        if region in ("lower_face", "lips"):
            # lower half of the face; lips ≈ middle band of lower half
            top = y + (bh // 2 if region == "lips" else bh // 3)
            mask[y + bh // 3:y + bh, x:x + bw] = 1
            mask[top:top + bh // 6, x:x + bw] = 1 if region == "lips" else 0
        elif region == "full_face":
            mask[y:y + bh, x:x + bw] = 1
        return mask

    # ------------------------------------------------------------------
    # 2. VISUAL MANIPULATION: region blend with controlled severity
    # ------------------------------------------------------------------
    def apply_visual_manipulation(self, frame_src: np.ndarray,
                                  frame_donor: np.ndarray,
                                  recipe: ForgeryRecipe):
        """Blend donor's facial region into src with severity-weighted alpha.

        NOTE: production-grade training replaces this stub with a viseme-aware
        model (e.g., a Wav2Lip-style encoder or a lightweight segmentation
        network). The stub keeps severity semantics consistent so the CML
        contrastive loss remains valid.
        """
        if recipe.visual_severity <= 0.0:
            return frame_src, np.zeros(frame_src.shape[:2], dtype=bool)

        boxes_src = self._detect_face_boxes(frame_src)
        boxes_donor = self._detect_face_boxes(frame_donor)
        if len(boxes_src) == 0 or len(boxes_donor) == 0:
            return frame_src, np.zeros(frame_src.shape[:2], dtype=bool)

        x, y, bw, bh = boxes_src[0]
        xd, yd, bwd, bhd = boxes_donor[0]
        src_box = frame_src[y:y + bh, x:x + bw]
        donor_box = frame_donor[yd:yd + bhd, x:xd + bwd]
        if donor_box.shape != src_box.shape:
            donor_box = cv2.resize(donor_box, (bw, bh))

        alpha = float(np.clip(recipe.visual_severity, 0.0, 1.0))
        # Gaussian feathering so the blend boundary mimics real face-swap art.
        mask = self._region_mask(frame_src, boxes_src[0], recipe.region)
        feather = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
        region_alpha = (feather * alpha)[..., None]

        out = frame_src.copy()
        blended = (src_box * (1 - alpha) + donor_box * alpha).astype(np.uint8)
        y0, y1 = max(0, y), min(frame_src.shape[0], y + bh)
        x0, x1 = max(0, x), min(frame_src.shape[1], x + bw)
        out[y0:y1, x0:x1] = (
            src_box[:y1 - y0, :x1 - x0] * (1 - region_alpha[:y1 - y0, :x0 - x1])
            + blended[:y1 - y0, :x0 - x1] * region_alpha[:y1 - y0, :x0 - x1]
        ).astype(np.uint8)

        # automatic GT: pixels actually replaced
        gt = ((region_alpha.squeeze() > 0.5) * (out != frame_src).any(-1))
        return out, gt

    # ------------------------------------------------------------------
    # 3. AUDIO MANIPULATION: swap + temporal offset with severity control
    # ------------------------------------------------------------------
    @staticmethod
    def apply_audio_manipulation(original_audio: np.ndarray,
                                 swapped_audio: np.ndarray,
                                 sample_rate: int,
                                 recipe: ForgeryRecipe) -> np.ndarray:
        """Cross-fade original with swapped track scaled by audio severity."""
        if recipe.audio_severity <= 0.0:
            return original_audio
        if swapped_audio is None:
            return original_audio

        n = min(len(original_audio), len(swapped_audio))
        a, s = original_audio[:n].astype(np.float64), swapped_audio[:n].astype(np.float64)
        alpha = float(np.clip(recipe.audio_severity, 0.0, 1.0))
        mixed = (1 - alpha) * a + alpha * s

        # inject lip-sync jitter: circular shift proportional to severity
        shift_samples = int(recipe.audio_offset_ms * sample_rate / 1000.0)
        return np.roll(mixed, shift_samples).astype(original_audio.dtype)
