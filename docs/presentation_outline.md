# Presentation outline

Seventeen slides for about fifteen minutes, plus backup. Slide text is in
English to match the report; the talk can be given in either language. Slide 14
is the one to drop if you are running short.

Each entry lists what goes on the slide, which figure to use, and what to say.
Figure paths are relative to the repository root. Numbers quoted here come from
`results/benchmark.csv` and the tables generated from it, except the regulariser
figures on slide 14, which come from `results/unregularised/` and
`results/lambda_sweep.csv`, and the training-variance figures on slide 16, which
come from `docs/model_selection.md`.

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

- At 25 measurements the best prior reaches 0.0194 against 0.0986 for the DCT
  baseline, **5.1 times better**, at a budget where neither baseline returns
  anything recognisable.
- Either baseline needs 400 measurements to reach about 0.011. All three VAE
  priors get there with **75**, a **5.3x** saving. The factor is the same
  against both baselines, so it does not rest on which one we pick. The paper
  reports 5 to 10x, so we land at the bottom of that range. The DCGAN at `k=30`
  needs 300, a 1.3x saving; the one at `k=20` never gets there.
- If asked about the pixel baseline: predicting a blank image scores 0.1178 on
  these digits, and the pixel baseline is at or above that at 10 and 25
  measurements, so a ratio against it there compares against nothing. At 400 it
  is mid-transition, median 0.0003 against mean 0.0107, which is why the count
  of images beaten differs so much between the two baselines.

---

## 11. What the reconstructions look like

**Slide.** `results/figures/reconstruction_grid.png`.

**Say.** The DCT baseline returns a noisy image until about 300 measurements
and the pixel one an almost blank image until about 400, while every
generative prior produces a plausible digit almost immediately. The DCGAN gives visibly
sharper strokes, because an adversarial generator is pushed towards samples a
discriminator accepts rather than towards the average of the plausible ones. But
sharper is not more accurate: a crisp digit of the wrong shape scores worse than
a slightly soft one of the right shape, and the table bears that out.

---

## 12. The ceiling

**Slide.** The flat part of the curves, with the floors, averaged from 300
measurements up.

| prior | error floor |
|---|---|
| VAE k=30 | 0.0070 |
| VAE, paper architecture | 0.0072 |
| VAE k=20 | 0.0076 |
| DCGAN k=30 | 0.0093 |
| DCGAN k=20 | 0.0119 |

**Have ready if asked about the DCGANs.** Each was trained on the same 54,000
images as the VAEs, and the epoch used was chosen among seven saved candidates on
held-out data. That choice earned about 11 per cent in both runs, since the error
does not fall monotonically with the epoch and the last one was not the best. One
seed each rather than the VAEs' two, because a run costs some fifteen hours.

**Say.** Past roughly 200 measurements the learned priors stop improving. The
bottleneck is no longer information, it is that the true digit is not in the
range of G, and that distance does not depend on the budget. From 500
measurements both baselines overtake everything, and at 750 the pixel one
recovers the digits almost exactly while the priors stay put. The paper says the
reversal takes more than 500 measurements; we see it from 500 onwards. A factor 1.7 separates the best floor
from the worst, so on this side of the plot the generator still matters, though
far less than the gap to the baselines does when measurements are scarce.

---

## 13. What survives the statistics

**Slide.** Three lines with their p-values.

**Say.** Every method sees the same images and the same matrices, so the
comparisons are paired and we test them that way.

- **The three VAE priors cannot be told apart.** Not one comparison among the
  paper's fully connected network, our k=20 and our k=30 reaches significance at
  any budget: every corrected p-value is 1. Their floors span 0.0005 and ten
  images cannot separate that. Say this plainly; do not rank them from the
  table.
- **Every prior beats both baselines at 100 measurements and loses to both at
  750.** That crossover is the reproduction and it is the claim that survives
  correction. Be precise if pressed: 50, 75, 100 and 300 are the budgets where it
  holds for all five priors at once, because the two DCGAN columns drop out
  elsewhere.
  Each VAE on its own is significantly better from 25 to 300.
- **The VAE beats the DCGAN at `k=20` at nine budgets out of ten, and at `k=30`
  only while measurements are scarce.** Every VAE has the lower mean error than
  every DCGAN at every budget; against `k=30` ten images are not enough to prove
  it beyond the low budgets. If asked whether the penalty is unfair to the
  GANs: we ran them at 0.001 too, the value the paper gives for a DCGAN, and it
  does not help, so the answer is no.

Saying out loud which differences do not reach significance is worth more than
claiming five results and defending three.

---

## 14. The latent regulariser

**Slide.** `results/figures/regularisation_comparison.png`.

**Say.** The paper uses $\lambda = 0.1$ and plots both variants, so we did the
same. It helps exactly where the measurements underdetermine the code: at 10
measurements it takes the best model from 0.0782 to 0.0680, a 13% gain. It costs
slightly once measurements are plentiful, 0.0054 to 0.0069 at 750, because the
same pull towards the prior keeps the solution off the closest point in the
range. The sweep confirms the mechanism: the norm of the recovered code falls
monotonically from 9.95 to 2.67, and the best value of lambda is not fixed but
falls with the budget, 1 at 10 measurements and 0 from 200 up. The single value
the paper recommends is a compromise, not an optimum at any one budget. Note
also that 0.1 is the value the paper gives for its MNIST VAE; for its DCGAN it
gives 0.001, which we do not use, so the DCGAN columns carry a penalty that was
never validated for them.

---

## 15. Cost

**Slide.** A small table.

| method | seconds to recover 10 images at one budget |
|---|---|
| Lasso, either basis | about 1.5 |
| VAE, paper architecture | 6.6 |
| VAE, convolutional | 15 and 19 |
| DCGAN | about 600 |

**Say.** A **93x** gap between the cheapest and the dearest learned prior.
Parameter count does not explain it: the DCGAN generator is only 4.5x larger, and
our convolutional decoder is smaller than the paper's yet twice as slow. What it
tracks is arithmetic per forward pass, and the DCGAN convolves 256 and 512
channels at nearly full resolution. The cost is paid at every reconstruction,
since recovery is itself an optimisation. And the punchline: the two most
expensive priors are also the two least accurate, at every budget.

---

## 16. A result about the method, not the models

**Slide.** The variance table from `docs/model_selection.md`.

**Say.** Training the same VAE with different random seeds gave generators whose
quality varied by a factor of **3.5**, comparable to the entire spread between
the architectures we set out to compare. The cause was partial posterior collapse:
the decoder leaning on a handful of latent directions and ignoring the rest,
which we measured directly. Adding a KL warm-up fixed it, and the spread across
seeds is a factor of 2.5, 1.3 and 1.2 for the three configurations. That
figure and the 3.5 are measured on different images and over a different
number of seeds, so do not present them as one ratio shrinking.

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
- The three VAE generators could not be told apart with ten test images, so the
  training run mattered more than the architecture, which is the one thing we
  can say about the choice between them.

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

**Which of the three VAE architectures is best?**
This experiment cannot say. With ten test images no comparison among them is
significant at any budget, and their error floors span 0.0005. If pressed on why
we cannot separate them: the sample is ten images, not that the models are
provably equal. More images would settle it, and the cost is recovery time.

**Ten test images is not many.**
Correct, and slide 13 says which comparisons survive and which do not. What
survives is the crossover: every prior beats both baselines at 50, 75, 100 and
300 measurements and loses to both at 750. What does not survive is any ranking
among the three VAEs, at any budget. Against the DCGAN at `k=20` the VAEs win at
nine budgets out of ten; against `k=30` only at the low budgets, though their
mean error is lower at every budget. The cost of
more images is the DCGAN recovery, about 600 seconds per budget.

**What does the theoretical guarantee actually require?**
That G is L-Lipschitz and that A is random Gaussian. It bounds the error
relative to the best reconstruction inside the range of G, so it does not
promise exact recovery, only that we approach the best the generator can do.

**Did you retrain everything?**
Yes. Six VAE runs and two DCGAN runs, each following the protocol and the
selection rule recorded in the repository, the DCGANs at some fifteen hours each
on CPU.

**Could you learn the measurement matrix?**
Yes, and it is the natural next step. The guarantee relies on the randomness of
A, so a learned matrix would trade the theory for empirical performance.
