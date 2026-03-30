from pathlib import Path
from typing import Callable

import pytest
from karpathy.microgpt import main as kmain
from simplemind.microgpt import main as smain


def emitter(vals: list[str]) -> Callable[[str], None]:
    def emit(val: str) -> None:
        vals.append(val)

    return emit


def test__microgpt__output_matches_reference_output() -> None:
    vals_s: list[str] = []
    vals_k: list[str] = []
    smain(10, emitter(vals_s))
    kmain(10, emitter(vals_k))
    assert vals_s == vals_k


@pytest.mark.slow
def test__precondition__reference_output_matches_known_reference_output() -> None:
    vals_k: list[str] = []
    kmain(100, emitter(vals_k))
    vals_ref = [s.strip() for s in (Path(__file__).parent / "ref_100_steps.txt").read_text().splitlines()]
    norm_vals_k = [k for k in [k.strip() for k in vals_k] if k]
    norm_vals_ref = [r for r in [r.strip() for r in vals_ref] if r]
    assert norm_vals_k == norm_vals_ref
