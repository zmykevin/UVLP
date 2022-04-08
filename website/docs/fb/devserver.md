---
id: devserver
title: 'Running MMF locally on Devservers'
sidebar_label: Running on devservers
---


## Running locally with BUCK

MMF code lives under `fbcode/faim/mmf` directory in fbcode. All the binaries available are in the `fbcode/faim/mmf/TARGETS` file. Below is the step by step tutorial on how to launch MMF Transformer on the [Hateful Memes](challenges/hateful_memes_challenge.md) dataset.

### Enable fwdproxy

Since MMF downloads some of the models and dependencies from internet, we first set
fwdproxy so as to enable intenet access for the devserver:

```bash
export http_proxy=fwdproxy:8080
export https_proxy=fwdproxy:8080
```

### Launch MMF using buck

We now launch `mmf:run` target as follows:

```bash
buck run @mode/dev-nosan faim/mmf:run -- \
    config=faim/mmf/projects/hateful_memes/configs/mmf_transformer/defaults.yaml \
    model=mmf_transformer \
    dataset=hateful_memes \
    run_type=train_val \
    env.data_dir=memcache_manifold://fair_mmf/tree/data
```

:::note

You need to set `@mode/dev-nosan` to access GPUs on your devserver. If you are running on a devserver without any GPUs, some of the models/functionalities will be slow.

:::

The command line opts are same as for OSS MMF. More details about config and command line opts can be found in the [configuration note](https://www.internalfb.com/intern/staticdocs/mmf/docs/notes/configuration).

On running this command, by default the experiment outputs like config, logs, model checkpoints etc will be saved in the `./save` directory. This can be overriden by adding the `env.save_dir=<path_to_your_save_dir>` to your command. This path can be a local path in your devserver or Manifold path or GFS path.


### Generate predictions

To generate predictions for a model on a dataset, use the `mmf:predict` target:

```bash
buck run @mode/dev-nosan faim/mmf:predict -- \
    config=faim/mmf/projects/hateful_memes/configs/mmf_transformer/defaults.yaml \
    model=mmf_transformer \
    dataset=hateful_memes \
    run_type=test \
    env.data_dir=memcache_manifold://fair_mmf/tree/data
```

which will return path to a prediction file. Note that we set `run_type=test` here to specify that we want to generate predictions on test set.

:::tip

Use `run_type=val` to generate predictions on `validation` set.

:::

### Running tests locally


To verify everything is working after your changes and to run tests on your devserver, run:

```bash
buck test faim/mmf:test
```

This will run all CPU tests and tests that do not require network access. In order to run GPU tests change the buck mode to `@mode/dev-nosan`, as:


```bash
buck test @mode/dev-nosan faim/mmf:test
```
