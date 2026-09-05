# Sample efficiency against Lasso at m = 400

Lasso reaches a per-pixel error of 0.0108 with 400 measurements.
Each learned prior matches or beats that level with:

| prior | measurements needed | speed-up |
|---|---|---|
| VAE, paper architecture, k=20 | 75 | 5.3x |
| VAE, k=20 | 100 | 4.0x |
| VAE, k=30 | 75 | 5.3x |
| DCGAN, k=20 | 200 | 2.0x |
| DCGAN, k=30 | never, in the sweep | n/a |
