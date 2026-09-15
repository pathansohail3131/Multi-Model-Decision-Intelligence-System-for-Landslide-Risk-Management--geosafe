"""Estimators, combination rules and model persistence.

The architecture trains one strong learner on each of four thematic feature
subsets plus one full-feature benchmark. Diversity comes from the data view,
not from mixing algorithms: the previous design mixed five algorithms on
identical features and the resulting stack underperformed its own best member,
because three of the five were tree ensembles making correlated errors.

Persistence in this package always serialises a model together with its fitted
preprocessing. A model that can be loaded without its transformer will
eventually be used without it, and the resulting predictions look plausible
while being wrong.
"""
