# Presentation outline

Seventeen slides for about fifteen minutes, plus backup. Slide text is in
English to match the report; the talk can be given in either language. Slide 14
is the one to drop if you are running short.

Each entry lists what goes on the slide, which figure to use, and what to say.
Figure paths are relative to the repository root. Every number quoted here comes from the
files under `results/` and from `docs/model_selection.md`, and each slide names
the one it draws on.

---

## 1. Title

Compressed Sensing using Generative Models. Both names, Politecnico di Milano,
Numerical Analysis for Machine Learning.

---

## 2. The problem

**Slide.** One equation, large:

$$y = A x^{\ast} + \eta, \qquad A \in \mathbb{R}^{m \times n}, \quad m \ll n$$

and one sentence: recover a 784 pixel image from 75 numbers.

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

## 5. The three priors

**Slide.**

| | trained by | architecture |
|---|---|---|
| VAE | maximising the ELBO | convolutional, ours |
| DCGAN | adversarial minimax game | convolutional, ours |
| VAE, paper | maximising the ELBO | fully connected 784-500-500-20 |

Baselines: Lasso in the pixel basis, which is what the paper uses on MNIST, and
Lasso in the DCT basis.

**Say.** The first two are our own designs, each trained at latent dimension 20
and 30. The third is the network Bora et al. actually use on MNIST. We included
it because our own models are not theirs, so without it a gap between our
numbers and theirs could be blamed either on the method or on the model, and we
could not tell which. Worth saying explicitly: the paper never runs a GAN on
MNIST, so that half of the comparison is our own extension.

---

## 6. Recovery

**Slide.** The objective

$$\hat z = \arg\min_z \lVert A G(z) - y \rVert_2^2 + \lambda \lVert z \rVert_2^2,
\qquad \hat x = G(\hat z)$$

with the residual curve and the reconstructed digit from
`notebooks/04_compressed_sensing_recovery.ipynb`.

**Say.** Gradient descent on the latent vector, not on the weights: the
generator is frozen and the only variable is twenty or thirty numbers. The
second term is the prior: since $z$ is Gaussian under both models,
$\lVert z \rVert^2$ is its negative log-likelihood. Point at the residual
falling by orders of magnitude and then flattening, and say that what remains is
not an optimisation failure. This sets up slide 12.

---

## 7. The objective is not convex

**Slide.** The four restarts figure from notebook 04, residual printed under
each reconstruction.

**Say.** G is a deep network, so different initialisations reach different local
minima. We run ten restarts and keep the smallest measurement error. Note two
things: the measurement error is computable from $y$ alone, without the ground
truth, so the rule is legitimate; and the penalty is deliberately excluded from
it, since ranking on the full objective would reward a small $\lVert z \rVert$
rather than a good reconstruction.

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
true image, not the measurement error the optimiser minimises. The latter falls
as m falls, simply because there are fewer constraints, so it says how well the
optimisation converged and not how good the answer is.

---

## 10. Results

**Slide.** `results/figures/error_vs_measurements.png`, full width.

**Say.**

- At 25 measurements the best prior reaches 0.0225 against 0.1255 for the
  paper's Lasso baseline, **5.6 times better**, at a budget where neither
  baseline returns anything recognisable.
- Lasso needs 400 measurements to reach 0.0108. The paper's architecture gets
  there with **75**, a **5.3x** saving. The paper reports 5 to 10x, so the
  reproduction lands at the bottom of their interval.
- Two baseline curves, not one: the paper uses the pixel basis on MNIST, the DCT
  basis is stronger below 400 measurements and much weaker above.

---

## 11. What the reconstructions look like

**Slide.** `results/figures/reconstruction_grid.png`.

**Say.** Both baselines return noise until about 200 measurements while every
generative prior produces a plausible digit almost immediately. The DCGAN gives visibly
sharper strokes, because an adversarial generator is pushed towards samples a
discriminator accepts rather than towards the average of the plausible ones. But
sharper is not more accurate: a crisp digit of the wrong shape scores worse than
a slightly soft one of the right shape, and the table bears that out.

---

## 12. The ceiling

**Slide.** The flat part of the curves, with the floors.

| prior | error floor |
|---|---|
| VAE, paper architecture | 0.0065 |
| VAE k=30 | 0.0074 |
| VAE k=20 | 0.0095 |
| DCGAN k=20 | 0.0098 |
| DCGAN k=30 | 0.0233 |

**Say.** Past roughly 200 measurements the learned priors stop improving. The
bottleneck is no longer information, it is that the true digit is not in the
range of G, and that distance does not depend on the budget. From 500
measurements Lasso overtakes everything, and at 750 it recovers the digits
almost exactly while the priors stay put. The paper says the reversal takes more
than 500 measurements and we find it there. A factor 3.6 separates the best floor
from the worst, so on this side of the plot the generator matters far more than
the recovery algorithm.

---

## 13. What survives the statistics

**Slide.** Three lines with their p-values.

**Say.** Every method sees the same images and the same matrices, so the
comparisons are paired and we test them that way.

- **The paper's simpler network beats ours.** Same latent dimension,
  significantly more accurate from 75 measurements up, p at or below 0.004,
  better on 9 of 10 images. We did not expect that.
- **k=30 beats k=20, but only from 75 measurements up**, p = 0.028 there and at
  or below 0.005 above. Below 75 the difference is not significant and its sign
  is not even stable: too few measurements to determine the extra coordinates.
- **The VAE beats the DCGAN at 10 and 50 measurements**, p = 0.011 and 0.024,
  but not at 25, 75 or 100; from 200 they are indistinguishable. Say that out
  loud rather than rounding it up to "the VAE wins when measurements are
  scarce".

Saying out loud which differences do not reach significance is worth more than
claiming five results and defending three.

---

## 14. The latent regulariser

**Slide.** `results/figures/regularisation_comparison.png`.

**Say.** The paper uses $\lambda = 0.1$ and plots both variants, so we did the
same. It helps exactly where the measurements underdetermine the code: at 10
measurements it takes the best model from 0.0808 to 0.0609, a 25% gain. It costs
slightly once measurements are plentiful, 0.0054 to 0.0064 at 750, because the
same pull towards the prior keeps the solution off the closest point in the
range. The sweep confirms the mechanism: the norm of the recovered code falls
monotonically from 7.7 to 2.5, and the best value of lambda is not fixed but
falls with the budget, 1 at 10 measurements and 0 from 200 up. The single value
the paper recommends is a compromise, not an optimum at any one budget.

---

## 15. Cost

**Slide.** A small table.

| method | seconds to recover 10 images at one budget |
|---|---|
| Lasso | 1.0 |
| VAE, paper architecture | 6.8 |
| VAE, convolutional | about 15 |
| DCGAN | about 496 |

**Say.** A **73x** gap between the cheapest and the dearest learned prior.
Parameter count does not explain it: the DCGAN generator is only 4.5x larger, and
our convolutional decoder is smaller than the paper's yet twice as slow. What it
tracks is arithmetic per forward pass, and the DCGAN convolves 256 and 512
channels at nearly full resolution. The cost is paid at every reconstruction,
since recovery is itself an optimisation. And the punchline: the most expensive
prior is also the least accurate at almost every budget.

---

## 16. A result about the method, not the models

**Slide.** The variance table from `docs/model_selection.md`.

**Say.** Training the same VAE with different random seeds gave generators whose
quality varied by a factor of **3.5**, larger than any difference between the
architectures we set out to compare. The cause was partial posterior collapse:
the decoder leaning on a handful of latent directions and ignoring the rest,
which we measured directly. Adding a KL warm-up fixed it, and the spread across
seeds fell to a factor 1.2.

The point to land: before this we would have reported that latent dimension 30
beats 20 by a wide margin, and that conclusion would have been an artefact of
comparing a lucky run against an unlucky one. Any comparison between priors is
only meaningful once it is larger than the variability of the procedure that
produced them. The selection rules were written down before the runs, and are in
the repository.

---

## 17. Conclusions

**Slide.** Three lines.

- A learned prior is worth about five times fewer measurements in the regime
  where measurements are scarce, which is the regime compressed sensing exists
  for, and that reproduces the reference paper.
- It buys that with a ceiling set by the generator, and with a reconstruction
  orders of magnitude more expensive.
- The simplest of the three generators was the best one, and the training run
  mattered more than the architecture.

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
- `results/figures/lambda_sweep.png`, the full sweep over the regulariser.
- The complete results table.

---

## Questions to be ready for

**Why are the entries of A scaled by one over the square root of m?**
So that A preserves norms in expectation. Otherwise the measurements shrink as
the budget grows and errors are not comparable across budgets.

**Why does Lasso overtake the generative priors?**
Representation error, slide 12. Lasso has no equivalent ceiling.

**Ten restarts and you keep the best. Is that not cheating?**
The selection uses the measurement error, computable from y alone. No ground
truth is involved, so the same rule works when the true image is unknown.

**Why is the paper's simpler network better than yours?**
We do not know, and we say so. Two candidate explanations: our decoder devotes
far fewer parameters to the output layer, and its spatial inductive bias may be
the wrong one for a manifold this simple. Separating them is future work.

**Ten test images is not many.**
Correct. It is enough for the paired comparisons to reach significance from 75
measurements up, and not enough below that, which is exactly what slide 13 says.
The cost is the DCGAN recovery, about 500 seconds per budget.

**What does the theoretical guarantee actually require?**
That G is L-Lipschitz and that A is random Gaussian. It bounds the error
relative to the best reconstruction inside the range of G, so it does not
promise exact recovery, only that we approach the best the generator can do.

**Did you retrain everything?**
The VAEs yes, with the procedure and the selection rule recorded in the
repository. The DCGANs no: about ten hours each on CPU. So the variance analysis
covers the VAEs only, and the same effect may well be present in the DCGAN
checkpoints.

**Could you learn the measurement matrix?**
Yes, and it is the natural next step. The guarantee relies on the randomness of
A, so a learned matrix would trade the theory for empirical performance.
