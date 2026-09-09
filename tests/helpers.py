"""Helpers shared by the tests that read the write-ups.

Two test modules need to know what a script does when it is run with no flags,
which is what the documented commands do. Reading the parser instead of running
it keeps the check cheap and avoids importing TensorFlow to learn an integer.
"""

from __future__ import annotations

import ast

from csgm.config import ROOT_DIR


def script_defaults(script: str) -> dict[str, object]:
    """Default value of every long option a script's parser declares.

    Args:
        script: File name in ``scripts/``, without the extension.

    Returns:
        Mapping from ``--flag`` to the default it is declared with. Flags whose
        default is computed rather than literal are absent.
    """
    tree = ast.parse((ROOT_DIR / "scripts" / f"{script}.py").read_text(encoding="utf-8"))
    found: dict[str, object] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"):
            continue
        name = next(
            (
                argument.value
                for argument in node.args
                if isinstance(argument, ast.Constant) and str(argument.value).startswith("--")
            ),
            None,
        )
        if name is None:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                found[name] = keyword.value.value
    return found
