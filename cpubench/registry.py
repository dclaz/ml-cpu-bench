"""Task registry — the single source of truth for tasks, sizes, and modes.

``@task`` appends to a deterministic ordered list. Run + report always iterate in this order.
Config expansion turns ``(mode, threads-leg)`` into concrete per-config records, expanding the
``engines`` of dual-engine tasks into separate scored entries ``name[engine]``.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from cpubench import CATEGORY_ORDER


@dataclass
class TaskSpec:
    name: str
    category: str
    func: Callable
    data: Callable  # builder(params, rng, engine) -> ctx.data (untimed)
    sizes: dict[str, dict]
    modes: frozenset[str]
    backend_sensitive: bool
    engines: tuple[str, ...] | None


_REGISTRY: list[TaskSpec] = []
_LOADED = False


def task(
    name: str,
    category: str,
    *,
    data: Callable,
    sizes: dict[str, dict],
    modes=("quick", "normal"),
    backend_sensitive: bool = False,
    engines: tuple[str, ...] | None = None,
):
    if category not in CATEGORY_ORDER:
        raise ValueError(f"unknown category {category!r}")

    def deco(func: Callable) -> Callable:
        _REGISTRY.append(
            TaskSpec(
                name=name,
                category=category,
                func=func,
                data=data,
                sizes=sizes,
                modes=frozenset(modes),
                backend_sensitive=backend_sensitive,
                engines=engines,
            )
        )
        return func

    return deco


def load_all_tasks() -> None:
    """Import the task modules so every ``@task`` registers (idempotent).

    After import, stable-sort by ``CATEGORY_ORDER`` so the registry's category blocks are
    deterministic regardless of module import order (within a category, declaration order is
    preserved by the stable sort). This is the single source of truth for run + report order.
    """
    global _LOADED
    if _LOADED:
        return
    importlib.import_module("cpubench.tasks")
    order = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
    _REGISTRY.sort(key=lambda spec: order[spec.category])
    _LOADED = True


def get_registry() -> list[TaskSpec]:
    return _REGISTRY


def get_task(name: str) -> TaskSpec:
    for spec in _REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(name)


def all_task_names() -> set[str]:
    load_all_tasks()
    return {spec.name for spec in _REGISTRY}


def scored_name(name: str, engine: str | None) -> str:
    return f"{name}[{engine}]" if engine else name


@dataclass
class RunConfig:
    """One concrete unit of work: a (task, engine, size, threads-leg) tuple."""

    task: str
    category: str
    engine: str | None
    mode: str
    params: dict
    threads: int
    threads_mode: str  # "all" | "single" | "explicit"
    cores: str  # "all" | "p" | "e"
    backend_sensitive: bool
    seed: int = 1337
    repeat: int = 5
    warmup: bool = True
    timeout: float = 3600.0
    cooldown: float = 2.0
    extra: dict = field(default_factory=dict)

    @property
    def scored_name(self) -> str:
        return scored_name(self.task, self.engine)

    @property
    def config_id(self) -> str:
        eng = self.engine or "none"
        raw = f"{self.task}__{eng}__{self.mode}__t{self.threads}__{self.threads_mode}__{self.cores}"
        return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)


def iter_task_specs(mode: str, tasks=None, exclude=None):
    """Yield (spec, engine) for tasks active in ``mode``, honouring --tasks/--exclude."""
    load_all_tasks()
    tset = set(tasks) if tasks else None
    xset = set(exclude) if exclude else set()
    for spec in _REGISTRY:
        if mode not in spec.modes:
            continue
        if tset is not None and spec.name not in tset:
            continue
        if spec.name in xset:
            continue
        engines = spec.engines if spec.engines else (None,)
        for engine in engines:
            yield spec, engine


def expand_configs(
    *,
    mode: str,
    legs: list[dict],
    tasks=None,
    exclude=None,
    seed: int = 1337,
    repeat: int = 5,
    warmup: bool = True,
    timeout: float = 3600.0,
    cooldown: float = 2.0,
) -> list[RunConfig]:
    """Cross every active (task, engine) with every threads-leg → ordered RunConfig list.

    ``legs`` is a list of dicts ``{threads, threads_mode, cores}`` (built by the controller from
    the CLI flags + detected topology).
    """
    configs: list[RunConfig] = []
    for spec, engine in iter_task_specs(mode, tasks=tasks, exclude=exclude):
        if mode not in spec.sizes:
            # fall back to the only size table present (defensive)
            params = next(iter(spec.sizes.values()))
        else:
            params = spec.sizes[mode]
        for leg in legs:
            configs.append(
                RunConfig(
                    task=spec.name,
                    category=spec.category,
                    engine=engine,
                    mode=mode,
                    params=dict(params),
                    threads=leg["threads"],
                    threads_mode=leg["threads_mode"],
                    cores=leg["cores"],
                    backend_sensitive=spec.backend_sensitive,
                    seed=seed,
                    repeat=repeat,
                    warmup=warmup,
                    timeout=timeout,
                    cooldown=cooldown,
                )
            )
    return configs
