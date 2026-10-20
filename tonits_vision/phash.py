"""64-bit perceptual hash, bit-compatible with the `imagehash` library's phash (hash_size=8).

Go compares these by Hamming distance. See DESIGN_REVIEW.md: on this screen, near-equal hashes are a
hint for staff, not proof that a screenshot was reused.
"""

import numpy as np
from PIL import Image
from scipy.fft import dct


def phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    size = hash_size * highfreq_factor
    pixels = np.asarray(image.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float64)
    coeffs = dct(dct(pixels, axis=0), axis=1)
    low = coeffs[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    value = int("".join("1" if b else "0" for b in bits), 2)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")
