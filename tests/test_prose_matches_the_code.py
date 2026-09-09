"""Check the numbers the write-ups take from the code, not from the results.

``test_reported_numbers.py`` compares the prose against ``benchmark.csv``. It
therefore reaches the results and nothing else, which leaves three large classes
of number with no guard at all: the layer sizes and parameter counts in the
architecture tables, the protocol constants the write-ups quote from the
scripts, and the ratios that summarise other numbers. A mutation audit over
every hand-written number showed these were the bulk of what nothing protected.

They are checked here against their source: the models are built and counted,
the parsers are read for their defaults, and every ratio is divided again.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from csgm.config import ROOT_DIR

REPORT = ROOT_DIR / "docs" / "report"
PRIORS = ["fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"]


def _defaults(script):
    """Every ``--flag`` default in a script, read from its source."""
    import ast

    tree = ast.parse((ROOT_DIR / "scripts" / f"{script}.py").read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            name = next(
                (
                    a.value
                    for a in node.args
                    if isinstance(a, ast.Constant) and str(a.value).startswith("--")
                ),
                None,
            )
            for keyword in node.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    found[name] = keyword.value.value
    return found


def _table_numbers(path, caption_marker):
    """The numbers of the table whose caption starts with ``caption_marker``."""
    text = (ROOT_DIR / path).read_text(encoding="utf-8")
    end = text.index(caption_marker)
    start = text.rindex(r"\begin{table}", 0, end)
    block = text[start:end]
    return [
        int(m.group(0).replace("{,}", ""))
        for m in re.finditer(r"\d{1,3}(?:\{,\}\d{3})+", block)
    ]


@pytest.fixture(scope="module")
def benchmark():
    path = ROOT_DIR / "results" / "benchmark.csv"
    if not path.exists():
        pytest.skip("benchmark.csv is not present")
    return pd.read_csv(path)


def test_the_architecture_tables_count_the_models_that_are_built():
    """Each table in the report must be the network the code produces.

    The tables are read by anyone reimplementing this, and nothing else in the
    suite looks at them, so a layer edited in ``csgm.models`` would leave them
    describing a network that no longer exists.
    """
    from csgm.models import (
        build_decoder,
        build_discriminator,
        build_encoder,
        build_fc_decoder,
        build_generator,
    )

    cases = [
        (build_generator(20), "docs/report/dcgan.tex", "caption{Structure of the DCGAN generator"),
        (
            build_discriminator(),
            "docs/report/dcgan.tex",
            "caption{Structure of the DCGAN discriminator",
        ),
        (build_encoder(20), "docs/report/vae.tex", "caption{Structure of the VAE encoder"),
        (build_decoder(20), "docs/report/vae.tex", "caption{Structure of the VAE decoder"),
    ]
    for model, path, marker in cases:
        stated = _table_numbers(path, marker)
        assert model.count_params() in stated, (
            f"{path}: the table totals {stated}, the built model has {model.count_params():,}"
        )
        for layer in model.layers:
            counted = layer.count_params()
            if counted >= 1000:
                assert counted in stated, (
                    f"{path}: layer {layer.name} has {counted:,} parameters, absent from the table"
                )

    # Both captions give the total for the other latent dimension in prose.
    dcgan = (REPORT / "dcgan.tex").read_text(encoding="utf-8")
    vae = (REPORT / "vae.tex").read_text(encoding="utf-8")
    assert f"{build_generator(30).count_params():,}".replace(",", "{,}") in dcgan
    assert f"{build_decoder(30).count_params():,}".replace(",", "{,}") in vae
    assert build_fc_decoder(20).count_params() > build_decoder(20).count_params(), (
        "the write-ups say our convolutional decoder is the smaller of the two"
    )


def test_the_protocol_the_write_ups_quote_is_the_scripts_defaults():
    """Running the documented commands has to reproduce the documented protocol.

    Every write-up states these numbers in prose, where no test reaches them,
    and the commands are given with no flags, so the defaults are the protocol.
    """
    benchmark = _defaults("run_benchmark")
    vae = _defaults("train_vae")
    dcgan = _defaults("train_dcgan")
    select = _defaults("select_dcgan")

    assert benchmark["--steps"] == 1000
    assert benchmark["--restarts"] == 10
    assert benchmark["--learning-rate"] == 0.01
    assert benchmark["--l2-penalty"] == 0.1
    assert benchmark["--noise-norm"] == 0.1
    assert benchmark["--n-images"] == 10

    assert vae["--epochs"] == 100
    assert vae["--batch-size"] == 100
    assert vae["--learning-rate"] == 0.001
    assert vae["--kl-warmup-epochs"] == 10
    assert vae["--validation-fraction"] == 0.1

    assert dcgan["--epochs"] == 50
    assert dcgan["--batch-size"] == 64
    assert dcgan["--learning-rate"] == 0.0002
    assert dcgan["--beta-1"] == 0.5
    assert dcgan["--generator-updates"] == 2
    assert dcgan["--checkpoint-every"] == 5
    assert dcgan["--checkpoint-from"] == 20
    assert dcgan["--validation-fraction"] == 0.1

    assert (select["--n-images"], select["--restarts"], select["--steps"]) == (32, 3, 500)

    saved = sorted(
        int(re.search(r"epoch(\d+)", path.name).group(1))
        for path in (ROOT_DIR / "models" / "dcgan_checkpoints").glob("*dim20*.keras")
    )
    if saved:
        expected = list(
            range(
                dcgan["--checkpoint-from"],
                dcgan["--epochs"] + 1,
                dcgan["--checkpoint-every"],
            )
        )
        assert saved == expected, "the write-ups describe seven candidates from the twentieth epoch"
        assert len(saved) == 7


def test_every_ratio_in_the_prose_divides_out(benchmark):
    """A ratio is right only if both values behind it are, and the pair is.

    These summarise numbers from three different sources, so none of them is
    reached by a check against ``benchmark.csv`` alone.
    """
    means = benchmark.pivot_table(index="m", columns="method", values="per_pixel_error")
    seconds = benchmark.groupby("method")["seconds_per_batch"].mean()
    floors = {k: means[k][means.index >= 300].mean() for k in PRIORS}

    assert round(400 / 75, 1) == 5.3
    assert round(400 / 300, 1) == 1.3
    assert round(means.loc[25, "lasso-dct"] / means.loc[25, PRIORS].min(), 1) == 5.1
    assert round(means.loc[25, "lasso"] / means.loc[25, PRIORS].min(), 1) == 6.1
    assert round(max(floors.values()) / min(floors.values()), 1) == 1.7
    assert round(seconds[PRIORS].max() / seconds[PRIORS].min()) == 93

    for latent_dim, spread in [(20, 1.28), (30, 1.42)]:
        record = ROOT_DIR / "models" / f"dcgan_selection_dim{latent_dim}.txt"
        if not record.exists():
            continue
        text = record.read_text(encoding="utf-8")
        scores = [float(v) for v in re.findall(r"epoch\d+\.keras: (0\.\d+)", text)]
        assert round(max(scores) / min(scores), 2) == spread
        last = float(re.search(r"last epoch[^\n]*at (0\.\d+)", text).group(1))
        assert round(100 * (last / min(scores) - 1)) == 11, (
            "the write-ups say the last epoch was about 11 per cent worse"
        )


def test_the_slide_figures_are_the_ones_the_notebooks_produced():
    """Slide assets copied out of a notebook must still be that notebook's output.

    Re-running a notebook rewrites its stored figure and nothing else, so
    without this the deck would go on showing a picture the repository no longer
    produces, exactly as the report's assets once did.
    """
    assets = ROOT_DIR / "docs" / "slides" / "assets"
    if not assets.exists():
        pytest.skip("the deck has no assets")

    stored = set()
    for notebook in sorted((ROOT_DIR / "notebooks").glob("*.ipynb")):
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        for cell in cells:
            for output in cell.get("outputs", []):
                payload = output.get("data", {}).get("image/png")
                if payload:
                    stored.add(hashlib.sha256(base64.b64decode(payload)).hexdigest())

    stale = [
        path.name
        for path in sorted(assets.glob("*.png"))
        if hashlib.sha256(path.read_bytes()).hexdigest() not in stored
    ]
    assert not stale, (
        f"{stale} are no longer any notebook's output; re-export them from the notebooks"
    )


def test_the_deck_covers_the_outline():
    """The deck and the outline have to describe the same talk."""
    deck = (ROOT_DIR / "docs" / "slides" / "slides.tex").read_text(encoding="utf-8")
    outline = (ROOT_DIR / "docs" / "presentation_outline.md").read_text(encoding="utf-8")

    titles = [m.group(1).strip() for m in re.finditer(r"^## \d+\. (.+)$", outline, re.M)]
    frames = re.findall(r"\\begin\{frame\}\{([^}]*)\}", deck)
    missing = [t for t in titles[1:] if t not in frames]  # the first is the title slide
    assert not missing, f"the deck has no frame for {missing}"
    assert len(re.findall(r"\\note\{", deck)) >= len(titles) - 1, (
        "every content slide carries what to say"
    )
    assert np.isclose(len(titles), 17), "the outline promises seventeen slides"
