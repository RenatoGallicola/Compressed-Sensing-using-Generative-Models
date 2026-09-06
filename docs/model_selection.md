# Model selection protocol

Written before running the experiments it describes, so that the rules cannot be
adjusted once the numbers are known. The commit that fixes each set of rules
precedes the commit that adds the checkpoints it governs, with one exception
stated at the top of the DCGAN section: the two GAN checkpoints currently in
`models/` predate these rules and do not follow them.

## Why this exists

Training the VAE repeatedly with the same script and different seeds produced
generators of very different quality. Measured as representation error, the best
reconstruction reachable inside the range of the generator:

| latent dim | seed 1337 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| 20 | 0.0268 | 0.0112 | 0.0254 | |
| 30 | 0.0358 | 0.0103 | 0.0122 | 0.0120 |

The spread within one latent dimension, a factor of 2.4 at `k=20` and 3.5 at
`k=30`, is larger than any difference between the two latent dimensions. The
diagnosis is partial posterior collapse: in the weaker runs the decoder relies
on a handful of latent directions and ignores the rest. Measuring how much each
latent coordinate moves the generated image, the ratio of the mean sensitivity
to the largest one is 0.31 for the best model and 0.05 for the worst, and that
ratio orders the models the same way the representation error does.

Those figures were measured on ten test digits, and they are what prompted the
intervention below. That is a decision informed by test-set behaviour, and it is
recorded here rather than left implicit. What it does not affect is which
checkpoint is used: no rule below consults the test split or the benchmark, so
no model was ever *chosen* on the data it is scored on. That is a statement about
selection. It is not a statement about training, and it does not cover the two
GAN checkpoints in use, which were trained on the test split before it was held
out.

## The intervention

KL warm-up: the KL term of the objective is scaled by a coefficient ramped
linearly from 0 to 1 over the first **10 epochs**. Early in training the encoder
is free to use the latent space without being pulled back towards the prior, and
the regulariser reaches full strength once the code is already informative. This
is the standard remedy for the failure mode diagnosed above.

The ramp length is fixed at 10 epochs and is not tuned, for any architecture.

## The rules for the VAE, fixed in advance

1. **Budget.** Two seeds, 1 and 2, for each of the convolutional `k=20`, the
   convolutional `k=30` and the fully connected `k=20` of the reference paper.
   Six runs. Every configuration gets the same budget: the same two seeds, the
   same warm-up length, the same 100 epoch limit with early stopping. No
   configuration receives extra search, and no further runs are added on the
   basis of the results.
2. **Selection.** For each configuration, keep the run with the lowest
   validation loss. Validation is evaluated with the KL coefficient at 1
   regardless of the warm-up schedule, so epochs and runs stay comparable.
3. **The test split is never seen.** Validation is a tenth carved out of the
   training split. The test split takes no part in training, in early stopping
   or in selection, which is what Bora et al. require of the generator.
4. **The benchmark is never consulted for selection.** Recovery results are
   computed only after the checkpoints are chosen.
5. **The outcome is reported as measured.**

## The rules for the DCGAN, fixed in advance

**The two GAN generators in `models/` do not follow any of the rules below.**
They were trained before these rules existed: on the training and test splits
together, with the optimiser settings the script used before it was aligned with
the reference paper, and with no selection among seeds or epochs. They are the
generators behind the two DCGAN columns of the published benchmark, which is why
those columns are reported with the limitation stated in the README rather than
presented as a clean held-out measurement. The rules below govern the retraining,
not the results as they stand.

A GAN has no likelihood, so none of rule 2 above transfers: there is no
validation loss, no stopping criterion, and sample quality oscillates from epoch
to epoch. Keeping whatever the last epoch produced is not a neutral default, it
is a choice made by the schedule. These rules therefore differ from the VAE's,
and where they differ the difference is stated.

1. **Budget.** One run for each of `k=20` and `k=30`, 50 epochs, using the only
   DCGAN recipe the reference paper gives (sec. 5.2): Adam at a learning rate of
   0.0002 with `beta_1 = 0.5`, mini-batches of 64, two generator updates per
   discriminator update. One seed rather than the VAE's two, because a run costs
   about ten hours of CPU against forty minutes for a VAE. This is a smaller
   budget than the VAE receives, and it is not hidden: the DCGAN columns are
   single draws from a distribution this document has itself shown to be wide.
2. **Training data.** The same 54,000 images the VAE trains on, so the two
   families see the same data and the held-out tenth is available to select on.
3. **Candidates.** The generator as saved every five epochs from epoch 20, seven
   candidates per run.
4. **Selection.** Lowest representation error over held-out images drawn from
   the training split, never the test split. Representation error is the floor
   every recovery result sits above and needs no discriminator, which makes it
   the one criterion available here that measures the thing the prior is for.
5. **The effect of this rule is published.** The representation error of every
   candidate is recorded, together with the spread across epochs and the value
   of the final epoch. A reader can therefore see how much work the selection is
   doing rather than be told it is harmless.
6. **The criterion differs from the VAE's, in the DCGAN's favour.**
   Representation error is a proxy for the quantity the benchmark reports, while
   the VAE is selected on the ELBO, which is not. Both are computed on held-out
   data and neither touches the test split, so neither leaks, but the asymmetry
   exists and runs towards the DCGAN. It is repeated wherever the two families
   are compared. As a check on its size, the same criterion is applied to the
   VAE seeds after the fact and the result records whether it would have chosen
   the same checkpoints as rule 2 did.
7. **The benchmark is never consulted for selection, and the outcome is reported
   as measured.**

## Why not simply keep the best model ever obtained

One checkpoint reached a representation error of 0.0064, better than anything
this procedure produced at the time. It came from a training run that predates
this repository and cannot be reproduced by the script here. Selecting it
because it scores well on the recovery benchmark would be selection on the test
measurement, which is exactly what the rules above exist to prevent.

## Outcome

Recorded after the runs finished. The rules above were not changed.

### VAE

Validation loss of the selected runs, all three configurations on the same
budget:

| configuration | selected seed | validation loss |
|---|---|---|
| convolutional `k=20` | 1 | 96.63 |
| convolutional `k=30` | 1 | 96.25 |
| fully connected `k=20` | 1 | 97.18 |

Rule 2 selects seed 1 in all three cases. The losses are comparable across
configurations, since the objective and the validation images are the same.

### Does the criterion matter

Rule 6 of the DCGAN section requires the size of the criterion asymmetry to be
measured rather than asserted. Applying the DCGAN's criterion to the VAE seeds
afterwards, on the same held-out training images, 32 of them:

| run | validation loss | representation error | selected by rule 2 |
|---|---|---|---|
| convolutional `k=20`, seed 1 | **96.63** | **0.00655** | yes |
| convolutional `k=20`, seed 2 | 106.70 | 0.01628 | |
| convolutional `k=30`, seed 1 | **96.25** | **0.00618** | yes |
| convolutional `k=30`, seed 2 | 99.76 | 0.00813 | |
| fully connected `k=20`, seed 1 | **97.18** | 0.00636 | yes |
| fully connected `k=20`, seed 2 | 97.32 | **0.00531** | |

The two criteria agree for both convolutional models and disagree for the fully
connected one, where the validation losses are separated by 0.14 and the
representation errors by 20 per cent. The reading is that validation loss orders
runs reliably when they are far apart and carries little information about
recovery quality when they are close.

The consequence is kept rather than undone: rule 2 selected a generator whose
representation error is 0.00636 when a run scoring 0.00531 was available. The
better model is not substituted, because choosing it now would be selection on a
criterion picked after seeing which answer it gives, which is what rules 3 and 4
exist to prevent. It also means the DCGAN's criterion is the better aligned of
the two with what the benchmark measures, so the asymmetry recorded in rule 6
runs in the DCGAN's favour by roughly the margin seen here.

### DCGAN

Not yet recorded. The runs the rules above describe have not been carried out,
and the generators currently in `models/` are the earlier ones described at the
top of that section. When the runs finish, the outcome comes from
`models/dcgan_selection_dim20.txt` and `models/dcgan_selection_dim30.txt`, which
`scripts/select_dcgan.py` writes, and the two DCGAN columns of the benchmark are
recomputed from the selected checkpoints.
