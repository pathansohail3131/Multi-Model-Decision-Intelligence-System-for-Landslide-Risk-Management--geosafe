"""Encoding conditioning factors into model features.

Three encodings in this package are not obvious and are justified at length in
``docs/methodology.md``:

* **Aspect is circular.** Encoded as ``sin``/``cos`` so that 359 deg and 1 deg
  are adjacent rather than maximally distant.
* **Flat cells have no aspect.** Cells with zero slope carry a placeholder
  aspect emitted by the terrain software; they are flagged and their aspect
  components zeroed rather than encoded as a real direction.
* **Lithology and LULC are categorical.** They are passed to the learner as
  categories, never label-encoded into a model that would read the codes as an
  ordering.

This package also owns the thematic feature groups that give the multi-model
architecture its diversity.
"""
