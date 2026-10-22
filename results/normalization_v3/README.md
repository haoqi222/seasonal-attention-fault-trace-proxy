# normalization_v3 — stage 3

Status: accepted.

Each fold's `norm_stats.json` contains 32-channel z-score means and population
standard deviations fitted only on that fold's eligible `train.npy` pixels.

The V2 feature tensor and stage-2 V3 masks are referenced by SHA-256 and were not
modified. No full normalized tensor, model checkpoint, or training result is included.
