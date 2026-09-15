"""The decision layers above susceptibility.

A probability per pixel is not something a planner can act on. This package
chains the components that turn a susceptibility surface into a prioritised
allocation of survey and mitigation effort:

1. a rainfall intensity-duration threshold component,
2. an exposure overlay built from roads and built-up land cover,
3. a prioritisation step combining susceptibility, confidence and exposure.

A ground-deformation layer (NISAR L-band InSAR) can be slotted in later as a
post-hoc activity multiplier. It is deliberately not a training feature:
observations postdate the inventory, so using them as a predictor would create
temporal leakage and force a retrain on every new acquisition.
"""
