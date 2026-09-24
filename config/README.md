# Configuration

| File | Tracked | Purpose |
|---|---|---|
| `config.yaml` | yes | Every machine-independent setting: grid, raster filenames, features, thematic partition, CV, imbalance, threshold costs, decision layers, inference and logging. |
| `config.local.yaml` | **no** (gitignored) | Machine-specific values, chiefly `paths.data_root`. Deep-merged over `config.yaml`. |
| `config.local.yaml.example` | yes | Template for the above. |

```bash
cp config/config.local.yaml.example config/config.local.yaml
# then set paths.data_root
python run_pipeline.py check
```

## Precedence

```
LANDSLIDE_DATA_ROOT  →  config/config.local.yaml  →  config/config.yaml
```

The merge is deep: overriding `paths.data_root` leaves the other `paths` keys
intact (`tests/test_config.py::test_merge_is_deep_not_shallow`). Lists are
replaced wholesale, not concatenated.

## Rules

- **No absolute path may appear under `src/`.** A test fails the build if one
  does; every path is either relative to the project root or derived from
  `data_root`.
- **No secrets belong here.** The project uses no API keys, tokens or
  credentials. `config.local.yaml` is gitignored because it is machine-specific,
  not because it is sensitive.
- HTTP-level settings (CORS origins) come from the environment instead — see
  [`../.env.example`](../.env.example).

`src/config.py` is the only module that resolves these files.
