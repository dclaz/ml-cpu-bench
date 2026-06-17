"""Import every task module so ``@task`` registration fires in fixed category order.

Order here MUST match ``cpubench.CATEGORY_ORDER`` so the registry (and every report) iterates
deterministically.
"""

from cpubench.tasks import (
    clustering,  # noqa: F401
    data_prep,  # noqa: F401
    factorization,  # noqa: F401
    linalg,  # noqa: F401
    models,  # noqa: F401
    sparse,  # noqa: F401
)
