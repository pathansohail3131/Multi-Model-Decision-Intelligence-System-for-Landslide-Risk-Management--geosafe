"""Command-line utilities that sit on top of the pipeline.

Run them as modules from the project root, e.g.::

    python -m scripts.predict sites.csv --output scored.csv

Training and evaluation are stages of the pipeline itself and have no separate
script: use ``python run_pipeline.py stage 4`` (train) and ``stage 3``/``stage 6``
(evaluation, interpretability). Duplicating them here would create a second
code path for results that must stay reproducible.
"""
