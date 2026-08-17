"""
SyncTrace — MediaPipe FaceMesh backend for region-aware pseudo-forgery
======================================================================
Provides precise facial landmark detection required by the
PseudoForgeryGenerator to build region masks (lips / lower_face / full_face)
from MediaPipe's 478 landmarks.

Region definitions (MediaPipe FaceMesh landmark indices, canonical tess):
  - lips:            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
                     321, 405, 314, 17, 84, 181, 91, 146 + inner contours
  - lower_face:      lips + chin region (indices 152, 234, 136, 172, 58, 132,
                     93, 361, 323, 454, 356 + jaw line 263..454 contour)
"""

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Canonical MediaPipe FaceMesh landmark groups
# ---------------------------------------------------------------------------
LIPS_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314,
    17, 84, 181, 91, 146,  # outer lip contour
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95,   # inner lip contour
]

CHIN_LANDMARKS = [152, 234, 136, 172, 58, 132, 93, 361, 323, 454, 356]
JAW_LANDMARKS = list(range(263, 455)) + [33, 7, 163]
NOSE_TIP = 1
LEFT_EYE_CENTER = 468
RIGHT_EYE_CENTER = 473


@dataclass
class FaceRegionBox:
    """Normalized (0..1) region boxes for a detected face."""
    x: float
    y: float
    w: float
    h: float
    lips_box: tuple[float, float, float, float]
    lower_face_box: tuple[float, float, float, float]
    landmarks: np.ndarray  # (N, 3) normalized coords


class FaceMeshDetector:
    """Wraps MediaPipe FaceMesh with temporal smoothing.

    Uses the lightweight face_detection model first (faster) and only runs
    the full 478-landmark mesh when a face is confidently detected.
    """

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        import mediapipe as mp  # lazy import: heavy dependency
        self.mp = mp
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> FaceRegionBox:
        """Detects the dominant face and returns region boxes in pixels."""
        h, w = frame.shape[:2]
        results = self.face_mesh.process(self.mp.cv2.cvtColor(
            frame, self.mp.cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([(p.x, p.y, p.z) for p in lm])  # (478, 3) normalized

        xs, ys = pts[:, 0], pts[:, 1]
        box = (xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min())

        lips = pts[LIPS_LANDMARKS]
        lower = pts[LIPS_LANDMARKS + CHIN_LANDMARKS]

        def norm_box(p):
            return (p[:, 0].min(), p[:, 1].min(),
                    p[:, 0].max() - p[:, 0].min(),
                    p[:, 1].max() - p[:, 1].min())

        return FaceRegionBox(
            x=box[0] * w, y=box[1] * h, w=box[2] * w, h=box[3] * h,
            lips_box=tuple(v * s for v, s in zip(norm_box(lips), (w, h, w, h))),
            lower_face_box=tuple(
                v * s for v, s in zip(norm_box(lower), (w, h, w, h))),
            landmarks=pts,
        )

    @staticmethod
    def ellipse_mask(h: int, w: int, box: tuple[float, float, float, float],
                     feather: int = 15) -> np.ndarray:
        """Smooth elliptical mask for a region box (x, y, w, h)."""
        mask = np.zeros((h, w), dtype=np.float32)
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        ax, ay = max(1, box[2] / 2), max(1, box[3] / 2)
        yy, xx = np.ogrid[:h, :w]
        norm = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
        mask[norm <= 1.0] = 1.0
        return mask
