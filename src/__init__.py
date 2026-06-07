"""MP2 sarcasm-detection package.

Public modules:

- :mod:`config`     - constants, paths, switches (single source of truth)
- :mod:`runtime`    - device, seeding, memory helpers
- :mod:`metrics`    - input building, metrics, run naming, logging
- :mod:`data`       - loading, leakage-safe splitting
- :mod:`eda`        - EDA tables and Fleiss' kappa
- :mod:`models`     - vocab, dataset, CNN/BiGRU, attention ladder
- :mod:`results`    - shared result store and row scaffold
- :mod:`runners`    - per-family experiment runners
- :mod:`experiment` - registries, the main loop, aggregation, stats
- :mod:`viz`        - plotting helpers
"""

__all__ = [
    "config",
    "runtime",
    "metrics",
    "data",
    "eda",
    "models",
    "results",
    "runners",
    "experiment",
    "viz",
]
