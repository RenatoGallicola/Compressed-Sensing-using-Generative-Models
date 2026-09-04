# Model selection protocol

Written before running the experiment it describes, so that the rules cannot be
adjusted once the numbers are known. The commit that introduces this file
precedes the commit that adds the resulting checkpoints.

## Why this exists

Training the VAE repeatedly with the same script and different seeds produced
generators of very different quality. Measured as representation error, the best
reconstruction reachable inside the range of the generator, over ten stratified
test digits:

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

## The intervention

KL warm-up: the KL term of the objective is scaled by a coefficient ramped
linearly from 0 to 1 over the first **10 epochs**. Early in training the encoder
is free to use the latent space without being pulled back towards the prior, and
the regulariser reaches full strength once the code is already informative. This
is the standard remedy for the failure mode diagnosed above.

The ramp length is fixed at 10 epochs here and is not tuned afterwards.

## The rules, fixed in advance

1. **Budget.** Two seeds, 1 and 2, for each of `k=20` and `k=30`. Four runs. No
   further runs are added on the basis of the results.
2. **Selection.** For each latent dimension, keep the run with the lowest
   validation loss. Validation is evaluated with the KL coefficient at 1
   regardless of the warm-up schedule, so epochs and runs stay comparable.
3. **The benchmark is never consulted for selection.** Recovery results are
   computed only after the checkpoints are chosen.
4. **The outcome is reported as measured.** If the resulting models are worse
   than the ones already in the repository, that is the result, and the earlier
   checkpoints are not silently reinstated.

## Why not simply keep the best model ever obtained

One checkpoint reaches a representation error of 0.0064, better than anything
this procedure has produced. It comes from a training run that predates this
repository and cannot be reproduced by the script here. Selecting it because it
scores well on the recovery benchmark would be selection on the test measurement,
which is exactly what rules 3 and 4 exist to prevent.
