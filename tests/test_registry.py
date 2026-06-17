"""Registry ordering + config expansion."""

from __future__ import annotations

from cpubench import CATEGORY_ORDER, registry


def test_registry_in_category_order():
    registry.load_all_tasks()
    cats = [spec.category for spec in registry.get_registry()]
    # categories appear as contiguous blocks in the fixed order
    seen = []
    for c in cats:
        if not seen or seen[-1] != c:
            seen.append(c)
    assert seen == [c for c in CATEGORY_ORDER if c in seen]
    # every category present
    assert set(seen) == set(CATEGORY_ORDER)


def test_long_runners_excluded_from_quick():
    registry.load_all_tasks()
    quick_names = {s.name for s in registry.get_registry() if "quick" in s.modes}
    for name in ("md_gpr", "md_lgbm_multi", "nlp_lda", "sp_lasso_cv"):
        assert name not in quick_names


def test_dual_engine_expands_to_two_entries():
    legs = [{"threads": 4, "threads_mode": "all", "cores": "all"}]
    configs = registry.expand_configs(mode="quick", legs=legs, tasks=["dp_groupby"])
    names = sorted(c.scored_name for c in configs)
    assert names == ["dp_groupby[pandas]", "dp_groupby[polars]"]


def test_expand_respects_exclude():
    legs = [{"threads": 4, "threads_mode": "all", "cores": "all"}]
    configs = registry.expand_configs(
        mode="quick", legs=legs, tasks=["la_gemm", "la_solve"], exclude=["la_solve"]
    )
    assert {c.task for c in configs} == {"la_gemm"}
