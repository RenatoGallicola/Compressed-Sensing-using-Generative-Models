# Sample efficiency at m = 400

For each prior, the smallest budget whose mean error is at or below the
level a baseline reaches with 400 measurements, and how many of the individual test images
it beats there. The comparison is given against both baselines, since the
speed-up depends on which one is used as the reference.

## Against Lasso (DCT basis)

Mean per-pixel error 0.0115 at 400 measurements (median 0.0113, largest 0.0151, 0 of 10 images below 0.001).

| prior | measurements needed | speed-up | images beaten |
|---|---|---|---|
| VAE, paper architecture, k=20 | 75 | 5.3x | 8 of 10 |
| VAE, k=20 | 75 | 5.3x | 8 of 10 |
| VAE, k=30 | 75 | 5.3x | 9 of 10 |
| DCGAN, k=20 | never, in the sweep | n/a | n/a |
| DCGAN, k=30 | 300 | 1.3x | 8 of 10 |

## Against Lasso (pixel basis)

Mean per-pixel error 0.0107 at 400 measurements (median 0.0003, largest 0.0834, 6 of 10 images below 0.001).

| prior | measurements needed | speed-up | images beaten |
|---|---|---|---|
| VAE, paper architecture, k=20 | 75 | 5.3x | 1 of 10 |
| VAE, k=20 | 75 | 5.3x | 1 of 10 |
| VAE, k=30 | 75 | 5.3x | 1 of 10 |
| DCGAN, k=20 | never, in the sweep | n/a | n/a |
| DCGAN, k=30 | 300 | 1.3x | 1 of 10 |

## The level of a blank image

Predicting an all-zero image scores 0.1178 per pixel on these digits, which is the error any method has to beat before it can be said to be reconstructing anything.

The pixel-basis baseline is at or above that level at m = 10, 25. A speed-up quoted against it at those budgets is a comparison against a blank image, so the DCT baseline is the meaningful reference there.

