# Sample efficiency against Lasso (DCT basis) at m = 400

That baseline reaches a mean per-pixel error of 0.0117 with 400 measurements (median 0.0124, largest 0.0169).
For each prior, the smallest budget whose mean error is at or below that level,
and how many of the individual test images it beats there.

| prior | measurements needed | speed-up | images beaten |
|---|---|---|---|
| VAE, paper architecture, k=20 | 50 | 8.0x | 8 of 10 |
| VAE, k=20 | 75 | 5.3x | 5 of 10 |
| VAE, k=30 | 75 | 5.3x | 7 of 10 |
| DCGAN, k=20 | 200 | 2.0x | 8 of 10 |
| DCGAN, k=30 | never, in the sweep | n/a | n/a |

## Why not Lasso (pixel basis)

At 400 measurements its error is 0.0108 on average but 0.0005 at the median, with 6 of 10 images already below 0.001 and the worst at 0.0842. It is midway through the transition from failure to near-exact recovery, so its mean at this budget is set by the digits it has not solved and is not a level worth comparing against.
