# Pre-trained checkpoints

Keras 3 (`.keras`) archives. At recovery time only the **generators** are used —
they are the prior $G$ that `csgm.recovery.recover` inverts. The discriminators
are kept so the adversarial training can be resumed or audited.

| file | role | latent dim | trained by |
|---|---|---|---|
| `vae_decoder_dim20.keras` | generator $G$ | 20 | `scripts/train_vae.py --latent-dim 20 --epochs 100` |
| `vae_decoder_dim30.keras` | generator $G$ | 30 | `scripts/train_vae.py --latent-dim 30 --epochs 100` |
| `gan_gen_dim20.keras` | generator $G$ | 20 | `scripts/train_dcgan.py --latent-dim 20 --epochs 50` |
| `gan_gen_dim30.keras` | generator $G$ | 30 | `scripts/train_dcgan.py --latent-dim 30 --epochs 50` |
| `gan_disc_dim20.keras` | discriminator $D$ | — | as above |
| `gan_disc_dim30.keras` | discriminator $D$ | — | as above |

All four generators map $z \in \mathbb{R}^k$ to a $28 \times 28 \times 1$ image
with sigmoid outputs in $[0, 1]$, and were trained on MNIST scaled to the same
range.

```python
from csgm.models import load_generator

G = load_generator("dcgan", 20)  # or load_generator("vae", 30)
```

## A note on the file format

These checkpoints were originally written by Keras 2 as HDF5 files carrying a
`.keras` extension. Keras 3 dispatches on the extension and expects a zip
archive, so it refused to open them (`Conv2DTranspose` also gained an
incompatible `groups` argument in the meantime). They were re-saved into the
current format by rebuilding the architectures in `csgm.models` and loading the
legacy weights topologically; outputs are bit-identical to the originals.
