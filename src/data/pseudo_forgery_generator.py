"""
SyncTrace — Controlled-Severity Pseudo-Forgery Generator
=========================================================
Generates identity-preserving pseudo-forgeries from authentic videos with a
CONTINUOUS severity label (0.0 .. 1.0), enabling:
  (a) self-supervised contrastive training without synthetic deepfake labels
  (b) severity regression (novel contribution)
  (c) automatic ground truth for localization evaluation (per-region masks)

Manipulation modes:
  1. VISUAL: region-level identity-preserving replacement (donor face region
     blended into the source frame with severity-controlled alpha)
  2. AUDIO:  cross-speaker audio swap with controlled temporal offset

Backends: "opencv" (Haar cascade, fast, coarse) or "mediapipe" (478-landmark
FaceMesh, precise region masks for lips / lower_face / full_face).
"""

import random
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ForgeryRecipe:
    """Describes one pseudo-forgery with controllable severity."""
    visual_severity: float = 0.0   # 0.0 = untouched, 1.0 = full region replaced
    audio_severity: float = 0.0    # 0.0 = original audio, 1.0 = swapped audio
    audio_offset_ms: int = 0       # extra lip-sync jitter in ms (0..500)
    region: str = "lips"           # "lips" | "lower_face" | "full_face"
    donor_video_path: str | None = None
    rng_seed: int | None = None

    @property
    def combined_severity(self) -> float:
        return min(1.0, self.visual_severity + self.audio_severity)


@dataclass
class GeneratedForgery:
    """Output of one generation step with automatic ground truth."""
    frames: np.ndarray            # (T, H, W, 3) BGR
    audio: np.ndarray             # raw waveform
    severity: float               # ground-truth combined severity
    visual_gt_masks: np.ndarray   # (T, H, W) bool — replaced pixels
    temporal_gt: np.ndarray       # (T,) bool — manipulated frames
    region: str
    recipe: ForgeryRecipe


def _feathered_region_mask(h: int, w: int, box: tuple[float, float, float, float],
                           region: str) -> np.ndarray:
    """Builds a smooth mask for the requested facial region box."""
    x, y, bw, bh = box
    mask = np.zeros((h, w), dtype=np.float32)
    if region in ("lips", "lower_face"):
        # lips ≈ upper-middle band of the lower face; lower_face = below nose
        if region == "lips":
            top, bottom = y + bh * 0.15, y + bh * 0.60
        else:
            top, bottom = y + bh * 0.45, y + bh
        mask[int(top):int(bottom), int(x):int(x + bw)] = 1.0
    elif region == "full_face":
        mask[int(y):int(y + bh), int(x):int(x + bw)] = 1.0
    else:
        raise ValueError(f"unknown region {region}")
    return cv2.GaussianBlur(mask, (21, 21), 0)


class PseudoForgeryGenerator:
    """Deterministic (seeded) pseudo-forgery generator with region masks."""

    BACKENDS = ("opencv", "mediapipe")

    def __init__(self, backend: str = "opencv", seed: int = 42):
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}")
        self.backend = backend
        self.rng = random.Random(seed)
        self._detector = None
        if backend == "mediapipe":
            from src.data.facemesh_backend import FaceMeshDetector
            self._detector = FaceMeshDetector()

    # ------------------------------------------------------------------
    # 1. FACE LOCALIZATION
    # ------------------------------------------------------------------
    def _face_box(self, frame: np.ndarray) -> tuple[float, float, float, float] | None:
        """Returns (x, y, w, h) in pixels or None.

        Uses cv2.FaceDetectorYN (OpenCV >= 4.7; Haar cascades were removed
        from OpenCV 5.x).
        """
        if self.backend == "opencv":
            if not hasattr(cv2, "FaceDetectorYN"):
                return None
            h, w = frame.shape[:2]
            if not hasattr(self, "_face_detector") or self._face_detector is None:
                model = cv2.data.haarcascades + "face_detection_yunet_2023mar.onnx"
                try:
                    self._face_detector = cv2.FaceDetectorYN.create(
                        model, "", (w, h), 0.6)
                except cv2.error:
                    return None
            self._face_detector.setInputSize((w, h))
            _, faces = self._face_detector.detect(frame)
            if faces is None or len(faces) == 0:
                return None
            x, y, bw, bh = faces[0][:4]
            return float(x), float(y), float(bw), float(bh)

        region = self._detector.detect(frame)
        if region is None:
            return None
        return region.x, region.y, region.w, region.h

    def _region_box(self, frame: np.ndarray, region: str):
        """Returns the region box via MediaPipe when available."""
        if self.backend == "mediapipe":
            r = self._detector.detect(frame)
            if r is None:
                return None
            boxes = {"lips": r.lips_box, "lower_face": r.lower_face_box,
                     "full_face": (r.x, r.y, r.w, r.h)}
            return boxes[region]
        return self._face_box(frame)

    # ------------------------------------------------------------------
    # 2. VISUAL MANIPULATION
    # ------------------------------------------------------------------
    def apply_visual_manipulation(self, frame_src: np.ndarray,
                                  frame_donor: np.ndarray,
                                  recipe: ForgeryRecipe) -> tuple[np.ndarray, np.ndarray]:
        """Blend donor's facial region into src with severity-weighted alpha.

        Returns (manipulated_frame, binary_gt_mask).
        """
        h, w = frame_src.shape[:2]
        if recipe.visual_severity <= 0.0:
            return frame_src, np.zeros((h, w), dtype=bool)

        box = self._region_box(frame_src, recipe.region)
        if box is None:
            return frame_src, np.zeros((h, w), dtype=bool)

        donor_box = self._region_box(frame_donor, recipe.region)
        if donor_box is None:
            return frame_src, np.zeros((h, w), dtype=bool)

        src_crop = self._crop_box(frame_src, box)
        donor_crop = self._crop_box(frame_donor, donor_box)
        donor_crop = cv2.resize(donor_crop, (src_crop.shape[1], src_crop.shape[0]))

        alpha = float(np.clip(recipe.visual_severity, 0.0, 1.0))
        # Feathered region mask keeps blending boundaries realistic.
        local_mask = np.ones(src_crop.shape[:2], dtype=np.float32)
        if recipe.region == "lips":
            local_mask = cv2.GaussianBlur(local_mask, (15, 15), 0)
        region_alpha = (local_mask * alpha)[..., None]

        blended = (src_crop * (1 - region_alpha) +
                   donor_crop * region_alpha).astype(np.uint8)

        out = frame_src.copy()
        x, y = int(box[0]), int(box[1])
        out[y:y + src_crop.shape[0], x:x + src_crop.shape[1]] = blended

        gt = ((local_mask * alpha) > 0.5)
        full_gt = np.zeros((h, w), dtype=bool)
        full_gt[y:y + gt.shape[0], x:x + gt.shape[1]] = gt
        return out, full_gt

    @staticmethod
    def _crop_box(frame: np.ndarray, box):
        x, y, bw, bh = box
        return frame[int(y):int(y + bh), int(x):int(x + bw)]

    # ------------------------------------------------------------------
    # 3. AUDIO MANIPULATION
    # ------------------------------------------------------------------
    @staticmethod
    def apply_audio_manipulation(original_audio: np.ndarray,
                                 swapped_audio: np.ndarray,
                                 sample_rate: int,
                                 recipe: ForgeryRecipe) -> np.ndarray:
        """Cross-fade original with swapped track scaled by audio severity
        and inject a lip-sync jitter proportional to severity."""
        if recipe.audio_severity <= 0.0 or swapped_audio is None:
            return original_audio

        n = min(len(original_audio), len(swapped_audio))
        a = original_audio[:n].astype(np.float64)
        s = swapped_audio[:n].astype(np.float64)
        alpha = float(np.clip(recipe.audio_severity, 0.0, 1.0))
        mixed = (1 - alpha) * a + alpha * s

        shift = int(recipe.audio_offset_ms * sample_rate / 1000.0)
        return np.roll(mixed, shift).astype(original_audio.dtype)

    # ------------------------------------------------------------------
    # 4. FULL CLIP GENERATION (video frames + audio)
    # ------------------------------------------------------------------
    def generate(self, src_frames: np.ndarray, src_audio: np.ndarray,
                 donor_frames: np.ndarray, donor_audio: np.ndarray,
                 sample_rate: int, recipe: ForgeryRecipe,
                 manipulate_frames_ratio: float = 0.6) -> GeneratedForgery:
        """Generate one pseudo-forgery clip with automatic GT.

        manipulate_frames_ratio: fraction of frames that receive the visual
        manipulation (the rest remain authentic, giving the evaluator a
        temporal localization signal).
        """
        t = len(src_frames)
        frames, gt_masks = [], []
        temporal_gt = np.zeros(t, dtype=bool)

        # deterministic frame selection for manipulation
        if recipe.rng_seed is not None:
            rng = random.Random(recipe.rng_seed)
        else:
            rng = self.rng
        n_manip = max(1, int(t * manipulate_frames_ratio))
        manipulated = sorted(rng.sample(range(t), n_manip))
        temporal_gt[manipulated] = True

        # pick donor frames cycling through the donor clip
        donor_idx = [i % len(donor_frames) for i in range(t)]

        for i in range(t):
            frame = src_frames[i].copy()
            mask = np.zeros(frame.shape[:2], dtype=bool)
            if i in manipulated:
                frame, mask = self.apply_visual_manipulation(
                    frame, donor_frames[donor_idx[i]], recipe)
            frames.append(frame)
            gt_masks.append(mask)

        audio = self.apply_audio_manipulation(
            src_audio, donor_audio, sample_rate, recipe)

        return GeneratedForgery(
            frames=np.stack(frames), audio=audio,
            severity=recipe.combined_severity,
            visual_gt_masks=np.stack(gt_masks),
            temporal_gt=temporal_gt, region=recipe.region, recipe=recipe,
        )

    # ------------------------------------------------------------------
    # 5. DATASET-LEVEL BUILDER (batch generation with recipe grid)
    # ------------------------------------------------------------------
    @staticmethod
    def default_recipe_grid(n_per_clip: int = 3,
                            severities=(0.0, 0.3, 0.6, 1.0),
                            regions=("lips", "lower_face"),
                            with_audio_swap: bool = True):
        """Produces a deterministic grid of ForgeryRecipes for a clip pair."""
        recipes = []
        for sev in severities:
            for region in regions:
                # visual-only recipes
                recipes.append(ForgeryRecipe(visual_severity=sev, region=region))
                # audio-only recipes (severity on audio side)
                if with_audio_swap:
                    recipes.append(ForgeryRecipe(audio_severity=sev))
                # combined recipe for the highest severity
                if sev == severities[-1]:
                    recipes.append(ForgeryRecipe(
                        visual_severity=sev, audio_severity=sev, region=region))
        return recipes[:n_per_clip * len(severities) * len(regions)]
