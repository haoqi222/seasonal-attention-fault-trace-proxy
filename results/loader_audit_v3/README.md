# V3 loader integration audit — stage 4

Status: accepted.

The audit loads all four folds through `dataset_common_v3.py`, verifies training-only
normalization, tests deterministic and synchronized patch augmentation, and proves
validation/test sliding-window containment and full role-core coverage.

No model, optimizer, checkpoint, or training result is created in this stage.
