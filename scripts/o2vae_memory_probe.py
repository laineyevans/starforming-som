"""Probe O2-VAE memory, iteration time, and the 304-pixel encoder trace."""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import torch

from o2vae_adapted.models.factory import build_model


def trace_model(model, image_size, channels, device):
    shapes = []
    hooks = []
    for index, layer in enumerate(model.q_net.eq_layers):
        hooks.append(layer.register_forward_hook(
            lambda _module, _inputs, output, index=index: shapes.append(
                (index, tuple(output.tensor.shape) if hasattr(output, "tensor") else tuple(output.shape))
            )
        ))
    with torch.no_grad():
        model.q_net(torch.zeros(1, channels, image_size, image_size, device=device))
    for hook in hooks:
        hook.remove()
    print("encoder layer trace:", shapes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=304)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    for batch_size in (4, 8, 16):
        for encoder_channels in ([2, 3, 4, 4, 6, 8], [3, 4, 6, 6, 8, 12]):
            try:
                model = build_model(
                    args.channels,
                    encoder_channels=encoder_channels,
                    image_size=args.image_size,
                    align_loss=False,
                ).to(device)
                trace_model(model, args.image_size, args.channels, device)
                batch = torch.rand(batch_size, args.channels, args.image_size, args.image_size, device=device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                output = model(batch)
                output[1].mean().backward()
                elapsed = time.perf_counter() - start
                peak = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda" else float("nan")
                )
                print(f"batch={batch_size} encoder={encoder_channels} seconds={elapsed:.3f} peak_gib={peak:.3f}")
                del model, output, batch
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"OOM batch={batch_size} encoder={encoder_channels}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise


if __name__ == "__main__":
    main()
