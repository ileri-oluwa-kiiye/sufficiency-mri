from __future__ import annotations

from scipy import fft as sfft

# Number of threads per FFT. Keep 1 when running several worker processes.
FFT_WORKERS = 1


def set_fft_workers(n: int) -> None:
    """Set the number of threads used by every FFT in this package."""
    global FFT_WORKERS
    FFT_WORKERS = int(n)


def fft2c(x, axes=(0, 1)):
    """Image -> centred k-space over two axes."""
    shifted = sfft.ifftshift(x, axes=axes)
    k = sfft.fftn(shifted, axes=axes, norm="ortho", workers=FFT_WORKERS)
    return sfft.fftshift(k, axes=axes)


def ifft2c(k, axes=(0, 1)):
    """Centred k-space -> complex image over two axes."""
    shifted = sfft.ifftshift(k, axes=axes)
    x = sfft.ifftn(shifted, axes=axes, norm="ortho", workers=FFT_WORKERS)
    return sfft.fftshift(x, axes=axes)
