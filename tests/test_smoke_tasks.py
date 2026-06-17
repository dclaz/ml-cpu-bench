"""Primary correctness guard (SPEC §11): every task runs and returns a checksum in quick mode.

Dual-engine tasks run on both engines; engine outputs are NOT asserted equal (speed, not
agreement). Sizes are clamped tiny so the suite is fast.
"""

from __future__ import annotations

import pytest

from cpubench import registry
from tests.helpers import run_task_tiny

registry.load_all_tasks()


def _cases():
    cases = []
    for spec in registry.get_registry():
        for engine in spec.engines or (None,):
            cases.append(pytest.param(spec, engine, id=registry.scored_name(spec.name, engine)))
    return cases


@pytest.mark.parametrize("spec,engine", _cases())
def test_task_runs(spec, engine):
    stats = run_task_tiny(spec, engine, repeat=2)
    # ran the timed region twice (read-only, no reset) and produced stats
    assert len(stats["reps_s"]) == 2
    assert stats["median_s"] >= 0.0
    checksum = stats["checksum"]
    assert isinstance(checksum, dict) and checksum, f"{spec.name} returned no checksum"
