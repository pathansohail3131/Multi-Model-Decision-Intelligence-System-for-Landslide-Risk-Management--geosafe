"""FastAPI application package.

Layout:

``main.py``
    Application factory, middleware, exception handlers, static frontend mount.
``routers/``
    HTTP routes, one module per resource group.
``services/``
    Model loading and scoring; the only modules that touch model bundles.
``schemas/``
    Request validation and the conditioning-factor catalogue.
``core/``
    Settings and error types shared by the above.
"""
