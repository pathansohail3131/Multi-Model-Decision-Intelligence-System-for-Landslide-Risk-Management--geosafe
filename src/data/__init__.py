"""Loading and validating the delivered data assets.

This package treats the existing data as read-only input. The labelled samples
and the aligned conditioning-factor rasters were produced and validated by an
earlier phase of work and are never regenerated here; the job of this package
is to load them, assert that they are what they claim to be, and surface the
places where they are not.

The audit module is deliberately a script that produces a report rather than a
set of assertions buried in a loader, because several of the findings (the
lithology coverage gap, the flat-cell aspect convention) are limitations to be
documented rather than bugs to be silently patched.
"""
