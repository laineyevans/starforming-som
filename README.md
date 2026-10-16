# starforming-som
repo for mphys project on self organising map application to almagal star forming regions

## Autoencoder method

`run_pipeline.py` selects the autoencoder with `config.autoencoder_method`, which
defaults to `"existing"`. The existing convolutional autoencoder and all of its
filenames remain unchanged. Set it to `"o2vae"` to use the locally adapted
orientation-invariant VAE; its `embed_data` mean (`z_mu`) is passed to the
downstream SOM, GSOM, or SOUMAP after the existing `StandardScaler`.

The O2-VAE path is lazy and requires the upstream-tested environment (Python
3.9, `torch==1.13.0`, `e2cnn==0.2.1`, and `torchgeometry==0.1.2`, plus the
packages in `/raid/scratch/mevans/o2vae/requirements.txt`). No packages are
installed by this repository. The adapted copy in `o2vae_adapted/` is based on
upstream commit `4c70d589954a86d114aba2d27ff7111573ac669c4`: the E2SFCNN
six-layer schedule is unchanged, while the decoder uses a derived 19x19 base
and four exact stride-2 stages (`19 -> 38 -> 76 -> 152 -> 304`).

The expected encoder trace for a 304x304 input is
`304 -> 300 -> 150 -> 150 -> 75 -> 37 -> 33 -> 1`; run
`python scripts/o2vae_memory_probe.py` in the real O2-VAE environment to
confirm the installed e2cnn pooling behavior and report peak GPU memory.
The defaults use global dataset min-max normalization to `[0, 1]` for BCE,
an enabled circular mask, safe-minimum batch size 4, and conservative channel
widths. `o2vae_pad_to_odd` is intentionally disabled until the rotation sanity
check has been run.

When saving is enabled, existing results remain under
`/raid/scratch/mevans/autoencoder_results/`; O2-VAE results are saved in the
sibling directory `/raid/scratch/mevans/o2vae_autoencoder_results/` using
`encoded_data_o2vae.npy`, `losses_o2vae.npy`, `model_state_o2vae.pth`, and
`settings_o2vae.txt`. Premade O2-VAE results must likewise be selected from
the configured `premade_o2vae_autoencoder_results_dir`; no method falls back
to the other method's directory.

Examples:

```python
config.autoencoder_method = "existing"
config.use_premade_autoencoder_results = False
```

```python
config.autoencoder_method = "o2vae"
config.use_premade_autoencoder_results = False
config.save_autoencoder_results = True
```
