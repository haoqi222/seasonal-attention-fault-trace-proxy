# dataset_v3 — stage 2 fold labels and masks

Status: accepted.

This directory was generated from the accepted stage-1 spatial-fold manifest. It does not contain fitted normalization statistics or model outputs.

Each fold directory contains:

- `y.npy` / `label.tif`: 0 background, 1 positive, 2 ignore.
- `partition.npy` / `partition.tif`: 0 excluded/buffer, 1 train, 2 validation, 3 test.
- `train.npy`, `val.npy`, `test.npy`: eligible boolean evaluation masks.
- `fold_stats.json`: role counts and purge audit values.

All three internal buffers are 50 rows on a 100 m grid (5 km). Test labels match the frozen V2 labels; held-out fault-ID influence zones are ignored in development roles.
