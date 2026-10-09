import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import numpy as np
import os

def umap_test(encoded_data, preds, config):
    reducer = umap.UMAP(random_state=42, n_neighbors=20, min_dist=0.07)
    reducer.fit(encoded_data)
    embedding = reducer.transform(encoded_data)
    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert(np.all(embedding == reducer.embedding_))
    embedding.shape

    fig, ax = plt.subplots()
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=preds, cmap="plasma")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")
    ax.set_title("UMAP projection of dataset")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.savefig(os.path.join(config.run_dir, 'umap_test.png'))

def tSNE(encoded_data, preds, config):
    pca = PCA(n_components=2)
    encoded_pca = pca.fit_transform(encoded_data)

    perplexity = np.arange(20, 40, 5)
    divergence = []

    for i in perplexity:
        model = TSNE(n_components=2, init="pca", perplexity=i)
        reduced = model.fit_transform(encoded_data)
        divergence.append(model.kl_divergence_)
    
    plt.figure()
    plt.plot(perplexity, divergence, marker='x', linewidth=1, color='k')
    plt.xlabel("Perplexity Values")
    plt.ylabel("Divergence")
    plt.savefig(os.path.join(config.run_dir,'perplexity.png'))

    tsne = TSNE(n_components=2, perplexity=400, random_state=42)
    encoded_tsne = tsne.fit_transform(encoded_data)
    print(tsne.kl_divergence_)
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        encoded_tsne[:, 0],
        encoded_tsne[:, 1],
        c=preds,              # color by cluster
        cmap="plasma",        # use plasma colormap
        #s=20                  # optional: point size
    )

    ax.set_title("t-SNE visualization of dataset")
    ax.set_xlabel("First t-SNE")
    ax.set_ylabel("Second t-SNE")

    # Add colorbar to show cluster mapping
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")

    fig.savefig(os.path.join(config.run_dir, "tsne_test.png"))
    plt.show()