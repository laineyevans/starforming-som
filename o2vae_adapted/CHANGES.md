# Adapted O2-VAE modules

The files in this directory are adapted from `/raid/scratch/mevans/o2vae` at
commit `4c70d589954a86d114aba2d27ff7111573ac669c4`. Upstream is distributed
under the license and attribution terms in that repository.

Changes are intentionally limited to:

- keeping the upstream `E2SFCNN` encoder schedule unchanged;
- changing source-tree imports to package-relative imports;
- adding a decoder whose base map and number of stride-2 stages are derived
  from the requested image size (`19 -> 38 -> 76 -> 152 -> 304`);
- exposing a small local model factory used by the pipeline wrapper;
- retaining the upstream VAE mean embedding (`embed_data -> z_mu`) and
  Fourier alignment loss.
- adding the pipeline wrapper's `[N,C,H,W]` validation, global dataset
  min-max normalization to `[0, 1]` for BCE, optional circular masking, and
  ordered `shuffle=False` embedding extraction;
- adding configurable batch size, epochs, learning rate, beta, alignment,
  AMP, gradient accumulation, and the reserved odd-padding option;
- implementing `o2vae_pad_to_odd` as 304x304 input padding to 305x305,
  followed by a decoder output pad from 304x304 to 305x305, so the VAE
  assertion and BCE loss operate on matching 305x305 tensors;
- adding the memory/shape probe and dependency-aware smoke-test scaffolding.
- saving O2-VAE results in a sibling `o2vae_autoencoder_results` directory
  with method-specific filenames, separate from existing autoencoder results.

The upstream encoder's actual 304-pixel spatial trace is documented as
`304 -> 300 -> 150 -> 150 -> 75 -> 37 -> 33 -> 1`, subject to the exact
pooling rounding implemented by the installed e2cnn version. The memory probe
verifies this trace when e2cnn is available.
