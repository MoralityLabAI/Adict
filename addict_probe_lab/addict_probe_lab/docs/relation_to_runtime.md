# Relation to Runtime

This repo expects traces from the runtime harness. The minimal contract is:

- `ctx`: [64]
- `commit_state`: [16]
- `route_mask`: [32]
- `score`: [4]

A fixed window of length 32 gives a default feature tensor of width 116 before any extra summaries.

You can pad to 128 or append compact oracle/task summaries.
