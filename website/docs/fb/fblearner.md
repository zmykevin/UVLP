---
id: fblearner
title: 'Running MMF jobs on FBLearner'
sidebar_label: Running on FBLearner
---

MMF provides a utility script for running large scale hyperparameter sweeps on FBLearner. A grid search is run on all permutations for the values provided for each of the hyperparameters in the script. The dotlist overrides provided via MMF's configuration system allows to easily override any configuration parameter through this script. This script is created based on sweep scripts provided in FAIRSeq.

## Using FBLearner sweep scripts

To launch a sweep on FBLearner, we will use an example script from `tools/sweeps/fb_sweeps/`, and run:

```bash
cd  faim/mmf && \
bento console --file tools/sweeps/fb_sweeps/sweep_mmbt_political_ads_hive.py -- -- \
    -p <prefix_for_save_directory> \
    -t -1 -g <num_gpus_per_node> -n <num_nodes>  \
    --run-as-secure-group <your_fblearner_secure_group> \
    --entitlement <your_entitlement>
```

This will create a checkpoint directory by default in a user GFS directory. We are also adding Manifold checkpoint dir support which can be specified by adding the `--checkpoint_dir=<manifold_path>` to the launch command. Details about other parameters can be found in the file `faim/mmf/tools/sweeps/lib/__init__.py`. MMF internally sets fwdproxy for downloading the
dependencies from the internet, so you don't need to worry about that.

In the sweep script, specifically note the section that sets data and cache dir to MMF's manifold dir.

```py
hyperparam(
    "env.data_dir", "memcache_manifold://fair_mmf/tree/data"
),
hyperparam(
    "env.cache_dir", "memcache_manifold://fair_mmf/tree/torch/mmf"
),
```

:::tip

Follow [slurm](https://www.internalfb.com/intern/staticdocs/mmf/docs/tutorials/slurm) tutorial to understand the basic structure of these sweep scripts and how config overrides can be specified for a list of hyperparameters.

:::


## Features available with MMF FBLearner

### Torchelastic multinode DDP

MMF Fbleaner supports multi-GPU multi-node training using torchelastic. Training runs can be scaled up easily on multiple nodes and GPUs. In addition, torchelastic support helps training jobs to be executed in a fault tolerant and elastic manner.


### Metrics and Tensorboard

Metrics and Tensorboard visualizations will be available in the `Output` and `Tensorboard` tabs in your FBLearner run. To enable Tensorboard visualization of your run, add `training.tensorboard=True` to your config.


:::note

If `Tensorboard` tab doesn't show anything even after enabling `training.tensorboard=True` in config, click on `More Actions` and then `Launch Tensorboard` in the FBLearner UI.

:::
