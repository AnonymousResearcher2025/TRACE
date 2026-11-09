"""TRACE experiment harness.

The C++ core is exposed as ``trace_harness._trace`` (built by ``build.sh``).
"""
from . import _trace  # noqa: F401
from ._trace import (  # noqa: F401
    Catapult,
    DarthStop,
    DCacheConfig,
    DCacheEngine,
    DiskExecutor,
    HnswExecutor,
    LogStop,
    QVCacheConfig,
    QVCacheEngine,
    QVStrategy,
    SnakeConfig,
    SnakeEngine,
    TraceConfig,
    TraceEngine,
    VamanaExecutor,
    build_vamana,
    replay,
    replay_parallel,
)

PATH_NAMES = {0: "cold", 1: "resolve", 2: "seeded", 3: "cache_hit"}
