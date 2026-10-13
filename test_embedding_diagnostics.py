import os
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np

from embedding_diagnostics import run_embedding_diagnostics
from som_module import train_som


def main():
    rng = np.random.default_rng(42)
    n_sources, copies, dim = 300, 4, 1600
    centers = rng.normal(size=(3, dim)) * 3
    encoded, labels, filenames, raw_datas = [], [], [], []
    for source in range(n_sources):
        base = centers[source % 3] + rng.normal(size=dim)
        for suffix in ("", "[hflip]", "[vflip]", "[hvflip]"):
            encoded.append(base + rng.normal(scale=0.03, size=dim))
            labels.append(f"source_{source}_masked{suffix}")
            filenames.append(f"source_{source}_cont_image.fits")
            image = np.zeros((64, 64), dtype=np.float32)
            y, x = 32 + source % 5, 32 + source % 7
            image[y - 3:y + 4, x - 3:x + 4] = 1
            raw_datas.append(image)
    encoded = np.asarray(encoded, dtype=np.float32)
    raw_datas = np.asarray(raw_datas)
    config = SimpleNamespace(
        n_num=6, m_num=6, rough_epochs_num=1500, fine_epochs_num=1500,
        rough_learning_rate=.2, fine_learning_rate=.05, rough_sigma=2.5,
        fine_sigma=1.0, fine_lr_end=.005, fine_sigma_end=.5,
        som_fine_method="random", som_block_size=500, train_in_blocks_flag=True,
        diag_max_points=1200,
    )
    som, preds, _, _, _ = train_som(encoded, None, config)
    config.run_dir = tempfile.mkdtemp(prefix="embedding-diagnostics-")
    try:
        result = run_embedding_diagnostics(
            encoded, preds, labels, filenames, raw_datas,
            np.zeros(len(encoded)), config, som=som,
        )
        output = result["output_dir"]
        assert os.path.isfile(os.path.join(output, "summary.txt"))
        assert os.path.isfile(os.path.join(output, "cell_summary.csv"))
        pca_purity = result["purity"]["PCA"][10]
        assert np.nanmean(pca_purity[1]) <= np.nanmean(pca_purity[0]) + 1e-12
        assert pca_purity[4] < np.nanmean(pca_purity[1])
        assert result["flip_all_same_cell"] > 0.9
        print(f"diagnostics output: {output}")
        print("synthetic diagnostics assertions: PASS")
        print("generated files:", len(os.listdir(output)))
    finally:
        shutil.rmtree(config.run_dir)


if __name__ == "__main__":
    main()
