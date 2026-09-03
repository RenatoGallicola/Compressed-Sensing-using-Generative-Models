# Presentation outline

Fifteen slides for about fifteen minutes, plus backup. Slide text is in English
to match the report; the talk can be given in either language.

Each entry lists what goes on the slide, which figure to use, and what to say.
Figure paths are relative to the repository root. Every number quoted here comes
from `results/summary.md` and `results/benchmark.csv`.

---

## 1. Title

Compressed Sensing using Generative Models. Both names, Politecnico di Milano,
Numerical Analysis for Machine Learning.

---

## 2. The problem

**Slide.** One equation, large:

$$y = A x^{\ast} + \eta, \qquad A \in \mathbb{R}^{m \times n}, \quad m \ll n$$

and one sentence: recover a 784 pixel image from 100 numbers.

**Say.** The system is underdetermined, so it has infinitely many solutions and
recovery is impossible without an assumption about which solutions are
plausible. Everything that follows is about what that assumption should be.

---

## 3. The classical assumption: sparsity

**Slide.** The Lasso problem, and the DCT truncation strip from
`notebooks/03_lasso_baseline.ipynb`: the same digit from 784, 200, 100, 50, 25
and 10 coefficients.

**Say.** MNIST really is compressible in the DCT basis, 25 coefficients out of
784 still give a recognisable digit. But compressed sensing does not get to
choose which coefficients to keep, it has to discover their support from random
projections, and that is what costs measurements.

---

## 4. The idea: replace sparsity with a learned prior

**Slide.** Left, "few non zero coefficients". Right, "looks like a digit".
Below, `results/figures/prior_samples.png`.

**Say.** A trained generator maps a low dimensional latent vector to an image,
and its range is a learned model of what a digit looks like. That is a far
stronger statement than sparsity, and it is learned from data instead of
designed by hand. These samples are the entire hypothesis space of the method,
which will matter on slide 12.

---

## 5. The two priors

**Slide.** Two tables, one conceptual and one on scale.

| | trained by | used at recovery time |
|---|---|---|
| VAE | maximising the ELBO | the decoder |
| DCGAN | adversarial minimax game | the generator |

| | VAE decoder | DCGAN generator |
|---|---|---|
| project | dense to $14^2 \times 64$ | dense to $3^2 \times 128$ |
| upsample | one transposed conv | three transposed convs, 128 to 512 filters |
| parameters ($k=20$) | 282,177 | 2,921,473 |

**Say.** Two different training principles reaching the same object, a
differentiable map from a small latent space to image space. The VAE also gives
an encoder, the GAN does not, and that is exactly why recovery has to be posed
as an optimisation problem rather than a forward pass.

---

## 6. Recovery

**Slide.** The objective

$$\hat z = \arg\min_z \lVert A G(z) - y \rVert_2^2, \qquad \hat x = G(\hat z)$$

with the residual curve and the reconstructed digit from
`notebooks/04_compressed_sensing_recovery.ipynb`.

**Say.** Gradient descent on the latent vector, not on the weights: the
generator is frozen and the only variable is twenty or thirty numbers. Point at
the residual falling by orders of magnitude and then flattening, and say that
what remains is not an optimisation failure. This sets up slide 12.

---

## 7. The objective is not convex

**Slide.** The four restarts figure from notebook 04, residual printed under
each reconstruction.

**Say.** G is a deep network, so different initialisations reach different local
minima. We run ten restarts and keep the smallest measurement residual. Note
that the residual is computable from the measurements alone, without the ground
truth, so the selection rule is legitimate.

---

## 8. Why so few measurements are enough

**Slide.** If G is L-Lipschitz, of the order of $k \log L$ random Gaussian
measurements suffice to come within a constant of the best reconstruction
available in the range of G.

**Say.** The cost scales with the latent dimension k, not the ambient dimension
n. 784 pixels but only 20 or 30 degrees of freedom, so the measurement budget is
driven by 20 or 30.

---

## 9. Experimental protocol

**Slide.** Four bullets.

- Ten test digits, one per class.
- Same measurement matrix and same noise for every method at every budget.
- Noise with a fixed expected norm of 0.1, so budgets stay comparable.
- Error is the squared distance to the ground truth, per pixel.

**Say.** Insist on the last point. The quantity plotted is the error against the
true image, not the measurement residual the optimiser minimises. The residual
falls as m falls, simply because there are fewer constraints, so it says how
well the optimisation converged and not how good the answer is.

---

## 10. Results

**Slide.** `results/figures/error_vs_measurements.png`, full width.

**Say.**

- With 25 measurements the VAE with k=30 reaches 0.0250 against 0.1049 for
  Lasso, **4.2 times better**, at a budget where the Lasso reconstruction is not
  a digit at all.
- Lasso needs 400 measurements to reach 0.0117. The VAE with k=30 gets there
  with **75**, a saving of **5.3x**. The paper reports 5 to 10x, so we land at
  the bottom of their range.
- Do not skip the right hand side of the plot, it is the interesting part.

---

## 11. What the reconstructions look like

**Slide.** `results/figures/reconstruction_grid.png`.

**Say.** Lasso returns noise until about 200 measurements. The VAE with k=20
finds a plausible digit almost immediately but never sharpens it, and at 750
measurements it still returns a shape halfway between a 3 and a 5. The DCGAN
with k=20 is crisp from 25 measurements, because an adversarial generator is
pushed towards samples a discriminator accepts rather than towards the average
of the plausible ones.

---

## 12. The ceiling

**Slide.** The flat part of the curves, with the four plateau values.

| prior | error floor |
|---|---|
| VAE k=30 | 0.0068 |
| DCGAN k=20 | 0.0100 |
| DCGAN k=30 | 0.0204 |
| VAE k=20 | 0.0286 |

**Say.** Past roughly 200 measurements the learned priors stop improving. The
bottleneck is no longer information, it is that the true digit is not in the
range of G, and that distance does not depend on the budget. From 500
measurements Lasso overtakes everything, and at 750 it is **4.4 times** more
accurate than the best generative model. This is predicted in the paper, which
says the reversal takes more than 500 measurements, and we find it exactly
there. These four numbers are properties of the generators, so the way to
improve the method here is a better generator, not a better optimiser.

---

## 13. The latent dimension

**Slide.** The k=20 and k=30 curves isolated, for each family.

**Say.** For the VAE, k=30 wins at every budget from 50 upwards, by 2.6 to 4.7
standard errors, so the difference is real: more latent capacity, lower ceiling.
For the DCGAN, k=20 is ahead from 75 upwards and the sign is consistent, but the
gap never exceeds 1.9 standard errors over ten images, so we present it as a
trend and not as a result. Saying this out loud is worth more than claiming a
result we cannot support.

---

## 14. Cost, and the surprise

**Slide.** A small table.

| method | seconds to recover 10 images at one budget |
|---|---|
| Lasso | 1.4 |
| VAE | about 20 |
| DCGAN | about 790 |

**Say.** The DCGAN is about **38 times** more expensive than the VAE, and the
cost is paid at every reconstruction, since recovery is itself an optimisation.
The whole sweep took 4.5 hours. And the surprise: the DCGAN is also the less
accurate of the two. The best model overall is the VAE with k=30, best of the
five methods at seven budgets out of ten. Sharper is not the same as more
accurate, and a sharp digit of the wrong shape scores worse than a slightly
blurred one of the right shape.

---

## 15. Conclusions

**Slide.** Three lines.

- A learned prior is worth about five times fewer measurements in the regime
  where measurements are scarce, which is the regime compressed sensing exists
  for.
- It buys that with a ceiling set by the generator, and with a reconstruction
  orders of magnitude more expensive.
- Both effects, including the crossover past 500 measurements, reproduce what
  the reference paper reports on MNIST.

**Say.** Close on where this matters. Medical imaging, and any setting where a
single measurement is slow, expensive or harmful to the subject, and where the
reconstruction is done once.

---

## Backup slides

Keep these after the conclusions, unshown unless asked.

- Lasso reconstructions across budgets, from notebook 03.
- The two dimensional VAE latent space, `docs/figures/vae_latent_space_2d.png`.
- ELBO components during VAE training, from notebook 01.
- DCGAN samples after two epochs next to the fifty epoch checkpoint, from
  notebook 02, if asked how hard the GAN was to train.
- The full table of results.

---

## Questions to be ready for

**Why are the entries of A scaled by one over the square root of m?**
So that A preserves norms in expectation. Otherwise the measurements shrink as
the budget grows and errors are not comparable across budgets.

**Why does Lasso overtake the generative priors?**
Representation error, slide 12. Lasso has no equivalent ceiling.

**Ten restarts and you keep the best. Is that not cheating?**
The selection uses the measurement residual, computable from y alone. No ground
truth is involved, so the same rule works when the true image is unknown.

**What does the theoretical guarantee actually require?**
That G is L-Lipschitz and that A is random Gaussian. It bounds the error
relative to the best reconstruction inside the range of G, so it does not
promise exact recovery, only that we approach the best the generator can do.

**Why is the VAE blurry?**
The likelihood and the KL term push the decoder towards the mean of the
plausible reconstructions. It shows up here as a visibly soft digit, though
notably not as a worse error.

**Why train two latent dimensions?**
To measure the trade-off on slide 13 instead of guessing it.

**Could you learn the measurement matrix?**
Yes, and it is the natural next step. The guarantee relies on the randomness of
A, so a learned matrix would trade the theory for empirical performance.

**Ten test images is not many.**
Correct, and it is why slide 13 reports the DCGAN comparison as a trend rather
than a conclusion. The cost is the DCGAN recovery, about 790 seconds per budget,
so more images means proportionally more computation.
