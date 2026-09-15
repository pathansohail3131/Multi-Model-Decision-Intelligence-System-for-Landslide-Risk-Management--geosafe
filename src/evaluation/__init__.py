"""Cross-validation, metrics and the ablation study.

Landslide points cluster spatially, so a random train/test split places
neighbouring pixels on both sides of the split and inflates every score. This
package therefore treats spatial block cross-validation as the real evaluation
and random cross-validation as a diagnostic, run alongside purely to quantify
how much optimism the random split injects.

Metrics lead with AUC-PR rather than AUC-ROC. At 1:10.8 imbalance a model can
post a strong ROC curve while having no usable operating point at all.
"""
