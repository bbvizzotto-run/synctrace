import numpy as np
from src.data.dataset import _mel_spectrogram
w = (np.random.rand(32000) * 2 - 1).astype(np.float32)
m = _mel_spectrogram(w)
print("mel shape:", m.shape)
