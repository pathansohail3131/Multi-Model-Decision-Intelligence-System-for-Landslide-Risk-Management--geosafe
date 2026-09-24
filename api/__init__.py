"""HTTP prediction service around the trained landslide susceptibility bundles.

The application package lives in :mod:`api.app`. Nothing here re-implements any
modelling: every number served comes from the saved
:class:`~src.models.bundle.ModelBundle` artefacts, the shared combiners in
:mod:`src.models.combiners`, or the pipeline's own reports.
"""
