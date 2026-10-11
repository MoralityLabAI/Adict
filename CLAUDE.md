# Adict

The persistent-selection experiments live in `addict_probe_lab/addict_probe_lab`; the design and results are in `docs/mechinterp_project_spec.md` there. Run Python from the repo's `.venv`.

## GPU jobs

The GPU is a 16 GB laptop card shared with other projects and agents. Start every GPU job through vramorama so it waits for a VRAM lease instead of pushing the card into spill (allocations silently moved to system memory, which makes every job on the card crawl):

```bash
/c/Users/patri/AppData/Local/Programs/vramorama/vramorama.exe run --vram 10G --owner adict-<what> -- .venv/Scripts/python.exe -u <script> ...
```

Add `--detach --out <log>` for jobs that must outlive the session. Qwen3.5-4B-Base needs about 10 GB to measure and 11 GB to train an adapter; `scripts/run_g1_pilot.sh` already wraps its steps. Run `vramorama` to see what is on the card and `vramorama leases` for the queue.

## Commits

Commit at the end of every turn. Checkpoints, activation shards and generated datasets under `out/` are gitignored; reports and logs are committed.
