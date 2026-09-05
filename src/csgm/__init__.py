"""Compressed sensing with deep generative priors.

Reference implementation of Bora et al., *Compressed Sensing using Generative
Models* (ICML 2017), applied to MNIST with a DCGAN and a convolutional VAE and
benchmarked against a Lasso/DCT sparsity prior.

The public API is intentionally small:

>>> from csgm import gaussian_measurement_matrix, measure, recover
>>> A = gaussian_measurement_matrix(m=100, n=784, seed=0)
>>> y = measure(x, A, noise_std=0.01)          # doctest: +SKIP
>>> result = recover(generator, y, A, latent_dim=20)   # doctest: +SKIP
"""

from csgm.baselines import dct2_basis, dct_basis, lasso_recover
from csgm.config import IMAGE_SHAPE, MODELS_DIR, N_PIXELS, RESULTS_DIR
from csgm.data import load_mnist, sample_images
from csgm.measurements import gaussian_measurement_matrix, measure
from csgm.metrics import per_pixel_l2, psnr, reconstruction_error
from csgm.recovery import RecoveryConfig, RecoveryResult, recover

__all__ = [
    "IMAGE_SHAPE",
    "MODELS_DIR",
    "N_PIXELS",
    "RESULTS_DIR",
    "RecoveryConfig",
    "RecoveryResult",
    "dct2_basis",
    "dct_basis",
    "gaussian_measurement_matrix",
    "lasso_recover",
    "load_mnist",
    "measure",
    "per_pixel_l2",
    "psnr",
    "reconstruction_error",
    "recover",
    "sample_images",
]

__version__ = "1.0.0"
