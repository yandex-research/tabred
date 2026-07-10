# Paper code and experimental results

Code that was used for the experiments in the paper.

## Repository structure

- [`./exp`](./exp) all experiment logs are in this folder
- [`./bin`](./bin) scripts for launching the experiments
- [`./lib`](./lib) library, data loading, utilities 

## Environment

There are two environments: one for local development on machines without GPUs -
`tabred-env-local.yaml`, another for the machines with GPUs `tabred-env.yaml`.

To create the environment with all the dependencies run `micromamba create -f` with the env file of
choice (for example `micromamba create -f tabred-env.yaml` on a server with GPUs).

## Example

To reproduce results for the MLP on the maps-routing dataset.

1. Create an environment
2. Create dataset (run preprocessing script)
3. Run `export CUDA_VISIBLE_DEVICES=0` (or whatever device you like)
4. Run `python bin/go.py exp/mlp/maps-routing/tuning.toml --force` (force, deletes the existing outputs)
