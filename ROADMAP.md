# ROADMAP

Named follow-ups, none of them silently folded into this study:

- A float16 arm, where the bandwidth story and the tolerance
  both change and the frozen rules must be written again.
- An online (single-pass) softmax variant against the two-pass
  in-register kernel used here.
- A second operation with a different traffic shape (layer
  normalization) under the same three-arm design.
- A rows sweep (1024, 16384) to separate launch-bound from
  DRAM-bound behaviour at short rows.
- A clock-noted rerun: record GPU clocks beside every timed
  block so laptop power state becomes a measured covariate
  instead of a named threat.
