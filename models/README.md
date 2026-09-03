# Pre-trained checkpoints

Keras 3 (`.keras`) archives. At recovery time only the **generators** are used:
they are the prior $G$ that `csgm.recovery.recover` inverts. The discriminators
are kept so the adversarial training can be resumed or audited.

| file | role | latent dim | trained by |
|---|---|---|---|
| `vae_decoder_dim20.keras` | generator $G$ | 20 | `scripts/train_vae.py --latent-dim 20 --epochs 100` |
| `vae_decoder_dim30.keras` | generator $G$ | 30 | `scripts/train_vae.py --latent-dim 30 --epochs 100` |
| `gan_gen_dim20.keras` | generator $G$ | 20 | `scripts/train_dcgan.py --latent-dim 20 --epochs 50` |
| `gan_gen_dim30.keras` | generator $G$ | 30 | `scripts/train_dcgan.py --latent-dim 30 --epochs 50` |
| `gan_disc_dim20.keras` | discriminator $D$ | n/a | as above |
| `gan_disc_dim30.keras` | discriminator $D$ | n/a | as above |

All four generators map $z \in \mathbb{R}^k$ to a $28 \times 28 \times 1$ image
with sigmoid outputs in $[0, 1]$, and were trained on MNIST scaled to the same
range.

```python
from csgm.models import load_generator

G = load_generator("dcgan", 20)  # or load_generator("vae", 30)
```

## File format

The archives are in the Keras 3 format and load with the TensorFlow version
pinned in `requirements.txt`. Note that Keras dispatches on the file extension:
a `.keras` file holding HDF5 content, which is what Keras 2 used to write, is
rejected by Keras 3 with a confusing "not a zip file" error. If you hit that,
rebuild the architecture with `csgm.models` and load the weights from a copy of
the file renamed to `.h5`.
