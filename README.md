# Compressed Sensing using Generative Models

[![CI](https://github.com/RenatoGallicola/Compressed-Sensing-using-Generative-Models/actions/workflows/ci.yml/badge.svg)](https://github.com/RenatoGallicola/Compressed-Sensing-using-Generative-Models/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A1703.03208-b31b1b.svg)](https://arxiv.org/abs/1703.03208)

Reproduction and extension of **Bora, Jalal, Price & Dimakis, *Compressed Sensing
using Generative Models* (ICML 2017)** on MNIST: a DCGAN generator and a VAE
decoder are used as *learned priors* to reconstruct images from a handful of
random linear measurements, and benchmarked against the classical Lasso/DCT
sparsity prior.

> Course project for **Numerical Analysis for Machine Learning**, MSc in Computer
> Science and Engineering, Politecnico di Milano.
> The original write-up is in [`docs/NAML_project_report.pdf`](docs/NAML_project_report.pdf).

<p align="center">
  <img src="results/figures/error_vs_measurements.png" width="88%"
       alt="Reconstruction error against the number of measurements, for Lasso and four generative priors">
</p>

**A VAE prior recovers an MNIST digit from 75 random measurements about as
accurately as Lasso does from 400.** Past 500 measurements the ranking reverses,
because a generative prior can only ever return an image its generator is able
to produce.

---

## The problem

Compressed sensing reconstructs an unknown signal $x^{\ast} \in \mathbb{R}^n$ from far
fewer measurements than its dimension:

$$y = A x^{\ast} + \eta, \qquad A \in \mathbb{R}^{m \times n}, \quad \eta \sim \mathcal{N}(0, \sigma^2 I), \quad m \ll n.$$

With $m \ll n$ the system is underdetermined and has infinitely many solutions,
so recovery is only possible by assuming *structure*. Classical theory assumes
**sparsity in a fixed basis**, $x^{\ast} = \Psi\theta$ with few non-zero
$\theta_i$, and recovers $x^{\ast}$ with an $\ell_1$ program such as Lasso.

This project replaces that assumption with a much stronger, *learned* one: that
$x^{\ast}$ lies near the range of a trained generator $G : \mathbb{R}^k \to \mathbb{R}^n$.
Instead of "few active coefficients", the prior says "**looks like a digit**".

## The method

Recovery becomes a search in the latent space of the generator rather than in
signal space:

$$\hat z = \arg\min_{z \in \mathbb{R}^k} \; \lVert A\,G(z) - y \rVert_2^2 + \lambda \lVert z \rVert_2^2, \qquad \hat x = G(\hat z).$$

Since $G$ is a deep network the objective is non-convex, so it is minimised with
**Adam on $z$** (the generator weights stay frozen) from **several random
restarts**, keeping the restart with the smallest measurement residual. The
theoretical pay-off, and the reason the approach works with so few
measurements, is that the sample complexity scales with the *latent* dimension:
if $G$ is $L$-Lipschitz, $O(k \log L)$ Gaussian measurements suffice for an
$\ell_2/\ell_2$ recovery guarantee.

Two priors are trained on MNIST, each at $k = 20$ and $k = 30$:

| | prior | trained by | used at recovery time |
|---|---|---|---|
| **VAE** | convolutional encoder/decoder, diagonal Gaussian posterior | maximising the ELBO | the **decoder** is $G$ |
| **DCGAN** | strided conv generator + discriminator | adversarial minimax game | the **generator** is $G$ |

<p align="center">
  <img src="docs/figures/vae_decoder_architecture.png" width="42%" alt="VAE decoder architecture">
  <img src="docs/figures/dcgan_generator_architecture.png" width="42%" alt="DCGAN generator architecture">
</p>

The measurement matrix $A$ has i.i.d. $\mathcal{N}(0, 1/m)$ entries, which makes
it an approximate isometry in expectation ($\mathbb{E}\lVert Ax \rVert^2 = \lVert x \rVert^2$),
so errors in measurement space and signal space stay on the same scale as $m$
changes.

## Results

Ten test digits, one per class, recovered by every method from the same
measurement matrices and the same noise. Error is the squared distance to the
ground truth, per pixel; lower is better.

|   m | Lasso (DCT) | VAE k=20 | VAE k=30 | DCGAN k=20 | DCGAN k=30 |
|----:|------------:|---------:|---------:|-----------:|-----------:|
|  10 |      0.1539 |   0.0696 |   0.0782 |     0.1048 |     0.0798 |
|  25 |      0.1049 |   0.0401 |**0.0250**|     0.0623 |     0.0487 |
|  50 |      0.0850 |   0.0328 |**0.0153**|     0.0426 |     0.0321 |
|  75 |      0.0707 |   0.0336 |**0.0111**|     0.0245 |     0.0315 |
| 100 |      0.0560 |   0.0287 |   0.0158 | **0.0144** |     0.0270 |
| 200 |      0.0366 |   0.0319 |**0.0071**|     0.0105 |     0.0210 |
| 300 |      0.0206 |   0.0268 |**0.0067**|     0.0109 |     0.0219 |
| 400 |      0.0117 |   0.0311 |**0.0067**|     0.0099 |     0.0196 |
| 500 |  **0.0065** |   0.0267 |   0.0068 |     0.0093 |     0.0209 |
| 750 |  **0.0015** |   0.0265 |   0.0067 |     0.0094 |     0.0189 |

The full table, one row per method, budget and image, is in
[`results/benchmark.csv`](results/benchmark.csv).

### Three regimes

**Scarce measurements, up to about 100.** The learned priors win by a wide
margin. At 25 measurements the VAE with `k=30` is 4.2x more accurate than Lasso,
which at that budget returns pure streak noise. This is the regime compressed
sensing exists for, and it is where the prior matters most.

**Sample efficiency.** Lasso needs 400 measurements to reach a per-pixel error of
0.0117. The VAE with `k=30` reaches it with 75, a **5.3x** saving, and the DCGAN
with `k=20` with 200, a 2x saving. Bora et al. report 5 to 10x for their models,
so the better of ours lands at the bottom of that range.

**Abundant measurements, past 500.** Lasso overtakes every learned prior and
keeps improving, reaching 0.0015 at 750 measurements against 0.0067 for the best
generative model, a 4.4x reversal. Nothing is wrong with the optimisation: the
generative curves are flat because the reconstruction is confined to the range
of the generator, and the distance from a real digit to that range does not
depend on how many measurements are taken.

### The representation error is the ceiling

Averaged over budgets of 200 and above, the error settles at 0.0068 for the VAE
`k=30`, 0.0100 for the DCGAN `k=20`, 0.0204 for the DCGAN `k=30` and 0.0286 for
the VAE `k=20`. Those four numbers are properties of the generators, not of the
recovery algorithm, and they are what a better prior would have to improve.

<p align="center">
  <img src="results/figures/reconstruction_grid.png" width="95%"
       alt="One digit reconstructed by every method at every measurement budget">
</p>

The grid makes the same point visually. Lasso produces noise until roughly 200
measurements. The VAE `k=20` finds a plausible digit almost immediately but
never sharpens it, and at high budgets it still returns a shape halfway between
a 3 and a 5. The DCGAN `k=20` is crisp from 25 measurements on, because an
adversarial generator is pushed towards sharp samples rather than towards the
mean of plausible ones.

### The latent dimension

For the VAE, `k=30` is better than `k=20` at every budget from 50 upwards, by 2.6
to 4.7 standard errors, so the difference is real: the larger latent space
represents digits more faithfully and lowers the ceiling.

For the DCGAN the picture is different. `k=20` is ahead from 75 upwards, and the
sign is consistent, but over ten images the gap never exceeds 1.9 standard
errors. We report it as a trend rather than a result; separating the two would
need more test images.

### Cost

One reconstruction is not free. Averaged over the sweep, recovering ten images
with ten restarts and a thousand Adam steps takes 1.4 s with Lasso, about 20 s
with a VAE decoder and about 790 s with a DCGAN generator, a **38x** gap between
the two learned priors driven by the size of the generator. The whole sweep took
4.5 hours of CPU time. Where reconstructions are frequent and measurements are
cheap, that cost is decisive on its own.



## Repository layout

```
├── src/csgm/                  installable package -- all the logic lives here
│   ├── measurements.py          random Gaussian sensing operator
│   ├── recovery.py              latent-space optimisation (the core algorithm)
│   ├── baselines.py             Lasso in an orthonormal DCT basis
│   ├── metrics.py               per-pixel L2 error, PSNR
│   ├── data.py, viz.py          MNIST loading, plotting helpers
│   └── models/                  VAE, DCGAN, checkpoint loading
├── scripts/                   command-line entry points
│   ├── train_vae.py             train the VAE, save the decoder
│   ├── train_dcgan.py           train the DCGAN, save the generator
│   ├── run_benchmark.py         the full sweep -> results/benchmark.csv
│   └── make_figures.py          csv -> figures and summary tables
├── notebooks/                 narrated walkthrough (01 VAE, 02 DCGAN, 03 Lasso, 04 recovery)
├── models/                    pre-trained checkpoints (k = 20 and k = 30)
├── results/                   benchmark table, summary tables and figures
├── docs/
│   ├── report/                  LaTeX source of the report
│   ├── figures/                 architecture diagrams
│   └── NAML_project_report.pdf  the compiled report
└── tests/                     pytest suite covering the package
```

## Getting started

```bash
git clone https://github.com/RenatoGallicola/Compressed-Sensing-using-Generative-Models.git
cd Compressed-Sensing-using-Generative-Models

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

> **NumPy is pinned below 2.0.** The TensorFlow 2.16/2.17 wheels are built
> against the NumPy 1.x ABI and fail to import otherwise.

Reconstruct a digit from 100 measurements, 13% of its 784 pixels:

```python
from csgm import gaussian_measurement_matrix, load_mnist, measure, per_pixel_l2, recover
from csgm.models import load_generator

(_, _), (x_test, _) = load_mnist(flatten=True)
x_star = x_test[0]

G = load_generator("dcgan", latent_dim=20)
A = gaussian_measurement_matrix(m=100, n=784, seed=0)
y = measure(x_star, A, noise_std=0.01, seed=0)

result = recover(G, y, A, latent_dim=20)
print(f"per-pixel error: {per_pixel_l2(result.x_hat, x_star)[0]:.4f}")
```

## Reproducing the results

```bash
# 1. (optional) retrain the priors -- pre-trained checkpoints ship in models/
python scripts/train_vae.py   --latent-dim 20 --epochs 100
python scripts/train_dcgan.py --latent-dim 20 --epochs 50

# 2. sweep every prior over every measurement budget  (~3 h on CPU)
python scripts/run_benchmark.py --n-images 10 --steps 1000 --restarts 10

# 3. turn the raw table into figures and summary tables
python scripts/make_figures.py
```

`run_benchmark.py` writes `results/benchmark.csv`, one row per
(method, $m$, image), so the analysis can be redone without re-running the sweep.
Every random draw (measurement matrices, noise, latent initialisations and the
choice of test images) is derived from a single `--seed`.

The notebooks are committed **with their outputs**, so every plot is readable
straight from GitHub without installing anything. They were executed top to
bottom against the checkpoints and the benchmark table in this repository.

## Method notes

A few implementation choices determine what the numbers mean, so they are worth
stating explicitly.

**The measurement matrix is scaled by $1/\sqrt{m}$.** Entries are drawn i.i.d.
from $\mathcal{N}(0, 1/m)$, which makes $A$ an approximate isometry in
expectation. Measurement space and signal space then stay on the same scale as
the budget changes, and so does the effective noise level.

**Quality is measured against the ground truth.** Every curve reports
$\lVert \hat x - x^{\ast} \rVert^2 / n$, the metric used by Bora et al. The
measurement residual $\lVert A G(\hat z) - y \rVert$ is the quantity the
optimiser minimises, and it falls as $m$ shrinks simply because fewer
constraints remain to satisfy, so it is recorded as an optimisation diagnostic
and never as a score.

**Latent codes are initialised from $\mathcal{N}(0, I)$**, the prior the
generators were trained under. Starting much closer to the origin biases the
search towards the blurry centre of the latent space.

**Every method sees the same inputs.** At each budget the same measurement
matrix, the same noise draw and the same ten stratified test digits, one per
class, are handed to Lasso and to each generative prior, and the curves are
averages over those ten images. Following Bora et al., the noise vector has a
fixed expected norm of 0.1 at every budget, so the per-component standard
deviation is $0.1/\sqrt{m}$.

### Limitations

- **Representation error sets the floor.** Recovery can only return an image the
  generator can produce, so beyond a few hundred measurements the error stops
  improving no matter how much optimisation is thrown at it. A sparsity prior
  has no such ceiling.
- **MNIST is easy.** Digits are a low-dimensional, near-binary manifold; the
  gap over Lasso would narrow on richer datasets.
- **Recovery is expensive.** Each reconstruction runs 1000 gradient steps
  by 10 restarts through the generator, orders of magnitude slower than a
  single convex solve.

## References

1. A. Bora, A. Jalal, E. Price, A. G. Dimakis. *Compressed Sensing using
   Generative Models*. ICML 2017. [arXiv:1703.03208](https://arxiv.org/abs/1703.03208)
2. D. P. Kingma, M. Welling. *Auto-Encoding Variational Bayes*. ICLR 2014.
   [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
3. A. Radford, L. Metz, S. Chintala. *Unsupervised Representation Learning with
   Deep Convolutional Generative Adversarial Networks*. ICLR 2016.
   [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
4. I. Goodfellow et al. *Generative Adversarial Networks*. NeurIPS 2014.
   [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
5. E. J. Candès, J. Romberg, T. Tao. *Robust Uncertainty Principles: Exact Signal
   Reconstruction from Highly Incomplete Frequency Information*. IEEE Trans.
   Inf. Theory, 2006.

## Authors

**Renato Gallicola** and **Matteo Forlivesi**, Politecnico di Milano.

Released under the [MIT License](LICENSE). MNIST is distributed under the
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) license.
