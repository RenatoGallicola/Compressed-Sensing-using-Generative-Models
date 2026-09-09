"""Read the protocol out of the prose and check it against the code.

The other guards assert that the code is as intended. That is not the same as
asserting the write-ups describe it: a test comparing a script default to a
constant written in the test passes whatever the README says. A mutation audit
over every hand-written number made the gap measurable, so the quantities the
documents state about the setup are extracted from the documents here and
compared with the value the code actually produces.

Every pattern is anchored on the words around the number, and each quantity
declares how many times it must be found, so a pattern that stops matching after
a rewording fails instead of passing silently.
"""

from __future__ import annotations

import ast
import json
import re

import pytest

from csgm.config import ROOT_DIR

DOCUMENTS = [
    ROOT_DIR / "README.md",
    ROOT_DIR / "docs" / "model_selection.md",
    ROOT_DIR / "docs" / "presentation_outline.md",
    ROOT_DIR / "docs" / "slides" / "slides.tex",
    ROOT_DIR / "models" / "README.md",
    *sorted((ROOT_DIR / "docs" / "report").glob("*.tex")),
    *sorted((ROOT_DIR / "notebooks").glob("*.ipynb")),
]


def _defaults(script):
    """Every ``--flag`` default in a script, read from its source."""
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


def _prose(path):
    """The authored text of a document, with markup that hides numbers removed."""
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        text = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    else:
        text = path.read_text(encoding="utf-8")
    text = re.sub(r"\\(?:textbf|texttt|emph|mathrm)\{([^}]*)\}", r"\1", text)
    text = text.replace("\\(", " ").replace("\\)", " ").replace("{,}", ",")
    text = re.sub(r"[\\{}`*$]", " ", text)
    return re.sub(r"\s+", " ", text)


@pytest.fixture(scope="module")
def prose():
    return {path.name: _prose(path) for path in DOCUMENTS}


VAE_DOCUMENTS = ("vae.tex", "01_vae_training.ipynb")
DCGAN_DOCUMENTS = ("dcgan.tex", "02_dcgan_training.ipynb", "model_selection.md")


def _quantities():
    """(label, pattern, expected text, how many times it must appear, documents).

    The last field scopes a pattern to the documents that state that quantity,
    since the two families use the same words for different values: a pattern
    reading ``batch size of`` everywhere would compare the DCGAN's 64 with the
    VAE's 100 and call one of them wrong.
    """
    benchmark = _defaults("run_benchmark")
    vae = _defaults("train_vae")
    dcgan = _defaults("train_dcgan")
    select = _defaults("select_dcgan")
    training_images = int(60_000 * (1 - dcgan["--validation-fraction"]))

    return [
        (
            "images the generators train on",
            r"(?:same|on the|trains on) (\d{2},\d{3}) images",
            f"{training_images:,}",
            3,
            None,
        ),
        (
            "DCGAN learning rate",
            r"learning rate of (0\.\d+)",
            str(dcgan["--learning-rate"]),
            2,
            DCGAN_DOCUMENTS,
        ),
        ("DCGAN momentum", r"beta_?1 *=? *(0\.\d+)", str(dcgan["--beta-1"]), 2, DCGAN_DOCUMENTS),
        (
            "DCGAN batch size",
            r"(?:mini-?batches of|batch size of) (\d+)",
            str(dcgan["--batch-size"]),
            2,
            DCGAN_DOCUMENTS,
        ),
        (
            "DCGAN epochs",
            r"(?:for|runs for|was|,) (\d+) epochs",
            str(dcgan["--epochs"]),
            3,
            DCGAN_DOCUMENTS,
        ),
        (
            "VAE batch size",
            r"(?:mini-?batches of|batch size of) (\d+)",
            str(vae["--batch-size"]),
            1,
            VAE_DOCUMENTS,
        ),
        (
            "VAE learning rate",
            r"learning rate of (0\.\d+)",
            str(vae["--learning-rate"]),
            1,
            VAE_DOCUMENTS,
        ),
        ("VAE epochs", r"up to (\d+) epochs", str(vae["--epochs"]), 1, VAE_DOCUMENTS),
        ("recovery steps", r"--steps (\d+)", str(benchmark["--steps"]), 1, None),
        ("recovery restarts", r"--restarts (\d+)", str(benchmark["--restarts"]), 1, None),
        ("images scored", r"--n-images (\d+)", str(benchmark["--n-images"]), 1, None),
        (
            "expected noise norm",
            r"expected norm[^.]{0,30}?(0\.\d+)",
            str(benchmark["--noise-norm"]),
            3,
            None,
        ),
        (
            "images the selection scores",
            r"(\d+) (?:images )?held out of training",
            str(select["--n-images"]),
            1,
            None,
        ),
        ("ambient dimension", r"(\d{3})[ -]pixel", "784", 2, None),
    ]


def test_the_documents_quote_the_protocol_the_code_runs(prose):
    """Every documented setup number must be the one the scripts default to."""
    wrong, counts = [], {}
    for label, pattern, expected, minimum, documents in _quantities():
        found = 0
        for name, text in prose.items():
            if documents and name not in documents:
                continue
            for stated in re.findall(pattern, text, re.IGNORECASE):
                found += 1
                if stated.rstrip("0").rstrip(".") != expected.rstrip("0").rstrip("."):
                    wrong.append(f"{name}: {label} is stated as {stated}, the code uses {expected}")
        counts[label] = (found, minimum)

    assert not wrong, "; ".join(wrong)
    thin = {k: v for k, v in counts.items() if v[0] < v[1]}
    assert not thin, (
        f"these patterns no longer find what they guard, so they guard nothing: {thin}"
    )


def test_the_dataset_the_documents_describe_is_mnist(prose):
    """The dataset facts are quoted in several places and never checked."""
    facts = [
        ("training set size", r"training set consists of (\d{2},\d{3}) images", "60,000", 1),
        ("test set size", r"test set consists of (\d{2},\d{3}) images", "10,000", 1),
        ("image side", r"(\d{2}) ?(?:x|times) ?\d{2} pixels", "28", 1),
        ("features", r"total of (\d{3}) features", "784", 1),
        ("classes", r"There are (\d+) classes", "10", 1),
    ]
    wrong, thin = [], []
    for label, pattern, expected, minimum in facts:
        found = 0
        for name, text in prose.items():
            for stated in re.findall(pattern, text, re.IGNORECASE):
                found += 1
                if stated != expected:
                    wrong.append(f"{name}: {label} is {stated}, MNIST has {expected}")
        if found < minimum:
            thin.append(label)
    assert not wrong, "; ".join(wrong)
    assert not thin, f"these patterns find nothing any more: {thin}"
