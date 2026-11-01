# Place names

Display-only layers that name a location in Uttara Kannada. The models never see
them, and no score depends on them.

| File | Contents | Properties |
|---|---|---|
| `villages.geojson` | 982 census villages and towns | `village`, `taluk`, `town` |
| `taluks.geojson` | 11 taluk boundaries | `taluk` |
| `towns.geojson` | 11 taluk headquarters | `name` |

**Source:** [datta07/INDIAN-SHAPEFILES](https://github.com/datta07/INDIAN-SHAPEFILES),
folder `STATES/KARNATAKA`, commit `08490ea`, filtered to census district code 563.

**Licence:** MIT, Copyright (c) 2022, as stated in that repository's `LICENSE`.
The notice below applies to these three data files:

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions: The above copyright
> notice and this permission notice shall be included in all copies or
> substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
> WITHOUT WARRANTY OF ANY KIND.

**CRS:** EPSG:32643 (WGS 84 / UTM 43N), the project grid. The source is EPSG:4326.

**Processing** (`scripts/build_places.py`):

- Villages are simplified to 20 m and taluks to 40 m, and coordinates are rounded
  to 1 m. Boundaries are for naming and display, not survey or legal use, so a
  point close to a boundary can land in the neighbouring village.
- The source has 986 village records. Four are slivers under 1 m² that collapse at
  1 m precision and are dropped: Brahmur (Ankola), Bellambi (Yellapur), Bappalgundi
  (Mundgod) and Virnoli (Joida).
- The village name is `VILNAME11`, falling back to `VILNAM_SOI` in title case.
- A census suffix such as `(CMC)`, `(TMC)`, `(TP)`, `(CT)` or `(CMC + OG)` is
  removed from the name and recorded as `town: true`.
- Taluk `Supa` is shown as `Joida (Supa)`.
