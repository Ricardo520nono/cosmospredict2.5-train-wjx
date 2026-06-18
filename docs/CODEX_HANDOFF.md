# Codex Handoff

Goal: maintain and launch CosmosPredict2.5 training for ActionFollowingBench S1 without repeatedly rediscovering path/config issues.

## Current Stable Baseline

The stable baseline is:

```bash
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_family_balanced_8gpu.sh
```

This corresponds to the earlier working script:

```bash
bash /mnt/gyc/cosmos-predict2.5-CoRL/scripts/train_cosmos_8gpu_afb_s1_family_balanced_chunk16.sh
```

The public package version has the same training behavior, but reads code/weights/outputs from:

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
```

## Non-Negotiables

- Camera: head camera only.
- Family-balanced mix: `expert:pca:raw:random-feasible = 3:1:1:1`.
- Chunk size: 16.
- Action dim: 14.
- Batch default: 2 per GPU on 8 GPUs.
- WandB stays enabled by default.
- Outputs default to this package under `outputs/cosmos_train_output`.

## Before Launching

Run:

```bash
cd /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
bash scripts/preflight.sh
```

If preflight fails, fix that first. Most failures are path/import/config mismatches and should not require a cloud job submission.

## Editing Rules

- Edit files under `code/cosmos-predict2.5-CoRL`.
- Keep top-level `scripts/` as thin, readable launchers.
- Keep docs updated whenever changing tasks, paths, cameras, chunk size, action dim, checkpoint cadence, or model weights.
- Do not add old experiment scratch scripts or dirty debugging notes to the top-level package.

## GitHub Note

The local package includes model weights for direct training. Do not push `models/` or `outputs/` to GitHub as normal git files. For GitHub, push code, scripts, and docs; keep weights on `/mnt/public_ckp/...` or use Git LFS only if the repo owner explicitly wants that.
