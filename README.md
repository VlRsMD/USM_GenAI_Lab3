# Laborator 3: GAN on MNIST

A GAN with Dense layers that learns to generate handwritten digits from MNIST. It is built with TensorFlow 2 / Keras 3.

## Project layout

```
Lab3/
├── main.py                  # entry point: runs the whole lab (steps 0–4)
├── gan_mnist/
│   ├── config.py            # GANConfig: every hyperparameter and output path
│   ├── data.py              # step 0: load MNIST, scale to [-1, 1], flatten, batch
│   ├── models.py            # steps 1–2: generator and discriminator
│   ├── losses.py            # BCE losses for D and G
│   ├── trainer.py           # step 3: GANTrainer (train_step + epoch loop)
│   ├── visualization.py     # image grids and loss curves
│   └── utils.py             # seeding, logging, saving the loss history
├── tests/test_gan.py        # shape, range, loss and training-step checks
├── requirements.txt
└── pyproject.toml           # pytest settings
```

## Architecture

| Network | Layers | Output |
|---|---|---|
| Generator | z(100) → Dense(256) + LeakyReLU(0.2) → Dense(512) + LeakyReLU(0.2) → Dense(784, tanh) | image in [-1, 1] |
| Discriminator | x(784) → Dense(512) + LeakyReLU(0.2) → Dense(256) + LeakyReLU(0.2) → Dense(1, sigmoid) | P(real) |

The training setup:

- Loss is `BinaryCrossentropy`, with a separate `Adam(2e-4, beta_1=0.5)` optimizer for each network.
- `d_loss = BCE(1, D(x)) + BCE(0, D(G(z)))`.
- `g_loss = BCE(1, D(G(z)))`, computed with fresh noise.
- The defaults are 30 epochs and batch size 128. A 4×4 grid from fixed noise is saved every 5 epochs, and a final 5×5 grid is made from new noise.

## Setup and run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python main.py                  # full run, 30 epochs
python main.py --epochs 15      # shorter run
python main.py --show           # also open the figures
pytest                          # run the tests
```

On a CPU each epoch takes about 4–5 seconds, so the full run finishes in a few minutes.

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `samples/epoch_XXX.png` | 4×4 grid from the same fixed noise (epoch 1, then every 5 epochs) |
| `final_grid.png` | 25 digits from fresh noise (the lab deliverable) |
| `losses.png` | per-epoch D and G losses, with dashed equilibrium lines |
| `history.json` | the loss values |
| `models/*.keras` | the trained generator and discriminator |

## Reading the results

- **Equilibrium losses.** When D can't tell real from fake (D = 0.5), `d_loss = 2 ln 2 ≈ 1.386` and `g_loss = ln 2 ≈ 0.693`. The plot shows these as dashed lines.
- **Loss curves are not a quality metric.** G's loss depends on how strong D is at that moment, so judge progress from the sample grids.
- **Mode collapse.** If the grid shows many near-identical digits, or only a few digit classes, the generator has collapsed onto a few modes.
- **Choosing the digit.** A vanilla GAN can't produce a specific digit on request. That needs a conditional GAN (cGAN).

## Notes on the implementation

- The last partial batch is dropped (`drop_remainder=True`), and the noise is sized from the real batch. Every step therefore compares equal-sized real and fake batches.
- `train_step` is compiled with `@tf.function` for speed.
