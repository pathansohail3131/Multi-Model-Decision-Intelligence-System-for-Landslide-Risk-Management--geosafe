"""District-wide raster prediction.

The prediction grid is 3738 x 5554 = 20,760,852 pixels, of which 10,929,202
fall inside the district. A fully materialised multi-band float32 feature stack
is roughly 1.25 GB, which is what blocked the previous attempt at producing a
district map.

Inference here is windowed: each block is read from the source rasters,
encoded, predicted and written out before the next block is read, so peak
memory is bounded by the block size rather than by the district.
"""
