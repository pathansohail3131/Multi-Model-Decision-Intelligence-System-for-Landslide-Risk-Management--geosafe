# Prediction API

FastAPI service that scores one 30 m terrain cell with the trained landslide
susceptibility bundles.

```bash
pip install -r ../requirements.txt -r ../requirements-api.txt   # from the project root
python -m uvicorn api.app.main:app --host 127.0.0.1 --port 8000
```

- Interactive docs: <http://127.0.0.1:8000/docs>
- Endpoint reference: [`../docs/api.md`](../docs/api.md)
- Layout and design notes: [`../docs/architecture.md`](../docs/architecture.md)

## Layout

| Path | Role |
|---|---|
| `app/main.py` | Application factory, CORS, lifespan, static frontend mount |
| `app/dependencies.py` | Process-wide service singletons and FastAPI dependencies |
| `app/core/` | HTTP settings and the shared error types/handlers |
| `app/routers/` | Routes: `health`, `prediction`, `district` |
| `app/schemas/` | Request models and the conditioning-factor catalogue |
| `app/services/` | The only modules that touch model bundles |

## Rules this service keeps

1. **No second implementation of the model.** Scoring goes through
   `ModelBundle.predict_proba` and its attached calibrator; the blend,
   disagreement and confidence come from `src.models.combiners`; the class
   bands from `src.inference.raster`.
2. **Refuse rather than guess.** Bundles without a calibrator or threshold are
   not served; the error names the pipeline stage to run. Confidence is `null`
   when its Stage 5 reference values are missing.
3. **Reject the impossible, flag the unfamiliar.** Values outside physical
   limits are 422s; values outside the training range are scored and returned
   in `warnings`.

`tests/test_api.py::test_predict_matches_the_bundles_directly` enforces rule 1.
