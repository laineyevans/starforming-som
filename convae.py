from sklearn.cluster import KMeans
import os 
import numpy as np
from astropy.io import fits
from scipy import ndimage
from collections import defaultdict, Counter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
from sklearn_som.som import SOM
import torch.nn.functional as F
import math
from typing import Iterator
from skimage.morphology import closing
from skimage.measure import label
from scipy.stats import chisquare
import re
from sklearn.metrics import silhouette_samples
import pandas as pd

#
"""
parameters
"""
#
directory = '/raid/scratch/mevans/newdata'
date = '11.12.25'
sqrt_pixel_number = 305
times_noise = 1
batch_size = 8
epochs = 2 # for autoencoder
loops = 10000 # for som
learning_rate = 1e-4
cluster_no = 16
grid_size = 5 # for cluster image plots
vmin, vmax = 0, 1  # normalized image range
cluster_algo = 'som'  # kmeans, dbscan or som
add_flipped_images = False
normalisation_type = 'linear' # 'linear' or 'logarithmic'
do_island_seg = True
threshold = 0.03 #threshold for island segmentation, on a scale of 0 to 1, below this value is considered background
minimum_island_size = 150  # minimum number of pixels for an island to be kept
image_results_save = f'{date}_{cluster_algo}_epochs{epochs}_{cluster_no}clusters_{times_noise}noise_convae_result.png'
text_results_save = f'{date}_{cluster_algo}_epochs{epochs}_{cluster_no}clusters{times_noise}noise_convae_result.txt'
core_test_results_save = f'{date}_epochs{epochs}_{cluster_no}clusters{times_noise}noise_core_info.txt'
classification_result_save = f'{date}_epochs{epochs}_{cluster_no}clusters{times_noise}noise_classification_info.txt'
cluster_examples_dir = f'convae_{date}_{cluster_algo}_epochs{epochs}_{cluster_no}clusters_{times_noise}noise_{do_island_seg}island_{threshold}threshold'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
"""
functions and classes
"""
#

class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),  # 305x305 -> 153x153
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1), # 153x153 -> 77x77
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 77x77 -> 39x39
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),# 39x39 -> 20x20 
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # output in [0,1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded[:, :, :x.size(2), :x.size(3)]
        return decoded

def extract_id_from_filename(filepath):
    # get only the filename
    fname = os.path.basename(filepath)
    
    # match everything before "_cont"
    match = re.match(r'(.+?)_cont', fname)
    if match:
        return match.group(1)

def island_segmentation(
    data: np.ndarray,
    threshold: float,
    return_background: bool = False,
    minimum_island_size: int = minimum_island_size,
) -> Iterator[np.ndarray]:
    """
    Yield masks of "islands" in a 2D image above a threshold.

    Arguments:
        data {np.ndarray} -- 2D image array
        threshold {float} -- Pixels above this value are considered part of islands

    Keyword Arguments:
        return_background {bool} -- If True, also return the background mask (default: False)
        minimum_island_size {int} -- Minimum number of pixels for an island to be kept (default: 5)

    Returns:
        Iterator[np.ndarray] -- Each yielded mask is True for the island, False elsewhere
    """
    # Step 1: Threshold and clean the mask
    mask = closing(data > threshold)

    # Step 2: Label connected regions
    img_labels, no_labels = label(mask, return_num=True, connectivity=2)

    # Step 3: Loop through islands
    for i in range(no_labels + 1):  # include background if requested
        if i == 0 and not return_background:
            continue
        island_mask = img_labels == i
        if np.sum(island_mask) < minimum_island_size:
            continue
        yield island_mask

#
"""
data loading & preprocessing
"""
#
datas = []
selected_files = []
selected_labels = []
new_datas = []
normalised_datas = []

for entry in os.scandir(directory):  
    if entry.is_file():
        filename = entry.path
        if filename.endswith('.fits') and "G221.9605-01.9926" not in filename:
            hdul = fits.open(filename)
            image_data = hdul[0].data
            hdr = hdul[0].header
            noise = hdr['AGSTDRES']
            zoom_factor = sqrt_pixel_number / np.sqrt(np.size(image_data))
            
            normalised_data = image_data
            normalised_data = np.nan_to_num(normalised_data, nan=np.nanmin(normalised_data))
            normalised_data = ndimage.zoom(normalised_data, zoom_factor)
            if np.nanmin(normalised_data) <= 0:
                normalised_data -= np.nanmin(normalised_data)
            if normalisation_type == 'linear':
                normalised_data = normalised_data / np.nanmax(normalised_data)
            if normalisation_type == 'logarithmic':
                c_norm = 500 / np.log(1 + np.max(normalised_data))
                normalised_data = c_norm * (np.log(normalised_data + 1))
                normalised_data = np.array(normalised_data, dtype = np.uint8)
            normalised_datas.append(normalised_data)

            # add rotated images
            if add_flipped_images == True:
                nhflip_data = np.flipud(normalised_data)
                nvflip_data = np.fliplr(normalised_data)
                nhvflip_data = np.flip(normalised_data)
                normalised_datas.append(nhflip_data.astype(np.float32))
                normalised_datas.append(nvflip_data.astype(np.float32))
                normalised_datas.append(nhvflip_data.astype(np.float32))

            # remove low-noise pixels
            image_data[image_data <= times_noise * noise] = 0
            
            # set nans to minimum
            image_data = np.nan_to_num(image_data, nan=np.nanmin(image_data))
            
            # resize to consistent shape
            image_data = ndimage.zoom(image_data, zoom_factor)

            if np.min(image_data) <= 0:
                    image_data -= np.min(image_data)
            
            # normalize to [0,1]
            if normalisation_type == 'linear':
                if np.max(image_data) != 0:
                    image_data /= np.max(image_data)

            # logarithmic transformation
            if normalisation_type == 'logarithmic':
                c = 500 / np.log(1 + np.max(image_data))
                image_data = c * (np.log(image_data + 1))
                image_data = np.array(image_data, dtype = np.uint8)
            
            if np.all(image_data.all == 0):
                print('an image is empty - from preprocessing')
            
            # add to list of data
            if image_data.size == sqrt_pixel_number**2:
                datas.append(image_data.astype(np.float32))
                selected_files.append(filename)
            
            # add rotated images
                if add_flipped_images == True:
                    hflip_data = np.flipud(image_data)
                    vflip_data = np.fliplr(image_data)
                    hvflip_data = np.flip(image_data)
                    datas.append(hflip_data.astype(np.float32))
                    datas.append(vflip_data.astype(np.float32))
                    datas.append(hvflip_data.astype(np.float32))
                    for i in range(3):
                        selected_files.append(filename)
            hdul.close()

#
"""
island segmentation
"""
#
for sample_idx, img_flat in enumerate(datas):
    img_2d = img_flat.reshape((sqrt_pixel_number, sqrt_pixel_number))

    min_size = minimum_island_size

    while True:
        # build a combined mask
        combined_mask = np.zeros_like(img_2d, dtype=bool)
        for island_mask in island_segmentation(img_2d, threshold, minimum_island_size=min_size):
            combined_mask |= island_mask 

        if np.all(combined_mask == 0):
            min_size -= 5
            if minimum_island_size < 0:
                print('could not reduce island size enough')
                break
        else:
            break   

    island_data = img_2d * combined_mask
    island_vector = island_data
    new_datas.append(island_vector)
    
if do_island_seg == True:
    datas = np.array(new_datas)
else:
    datas = np.array(datas)
datas = datas[:, np.newaxis, :, :]  # add channel dimension (N,1,H,W)
print(f"Files inputted: {len(datas)}")

tensor_data = torch.tensor(datas)
dataset = TensorDataset(tensor_data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#
"""
convolutional autoencoder
"""
#
model = ConvAE().to(device)
loss_function = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

#
"""
training
"""
#
for epoch in range(epochs):
    epoch_loss = 0
    for (images,) in loader:
        images = images.to(device) # use multiple cores
        optimizer.zero_grad() # set gradients to 0
        reconstructed = model(images) # put the images into the model
        loss = loss_function(reconstructed, images) # find the loss
        loss.backward() # calculate gradients
        optimizer.step() # perform optimisation to update parameters
        epoch_loss += loss.item() * images.size(0) # add loss (times batch no. by averge loss)
    epoch_loss /= len(dataset) # at end of epoch, total summed loss / no. of samples, mean epoch loss
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}") # print losses as it goes

# 
"""
plot reconstructed images
"""
#
model.eval() # go into evaluation mode
with torch.no_grad(): # stops from computing gradients
    dataiter = iter(loader)
    (images,) = next(dataiter)
    images = images.to(device)
    reconstructed = model(images)
    all_data = torch.tensor(datas, dtype=torch.float32).to(device)
    encoded = model.encoder(all_data)   # pass through encoder, shape: (N, C, H, W)
    encoded_flat = encoded.view(encoded.size(0), -1)  # flatten the encoded image, → shape (N, C*H*W)
    encoded_np = encoded_flat.cpu().numpy() # move into a numpy array

num_images = min(len(images), 10)
fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))
for i in range(num_images):
    original = images[i, 0].cpu().numpy()
    recon = reconstructed[i, 0].cpu().numpy()
    diff = original - recon
    c_diff = 500 / np.log(1 + np.max(diff))
    diff_log = c_diff * (np.log(diff + 1))
    diff_log = np.array(diff_log, dtype = np.uint8)
    #print(f'difference max {np.max(diff)}')
    #print(f'difference min {np.min(diff)}')
    axes[0, i].imshow(original, vmin=np.min(original), vmax=np.max(original))
    axes[0, i].axis('off')
    axes[1, i].imshow(recon, vmin=np.min(recon), vmax=np.max(recon))
    axes[1, i].axis('off')
    axes[2, i].imshow(diff_log, cmap='berlin', vmin=-abs(np.max(diff_log)), vmax=np.max(diff_log))   
    axes[2, i].axis('off')
plt.savefig(image_results_save)
plt.close(fig)

n_samples = min(3, len(images))
images_cpu = images.cpu().detach()
recon_cpu = reconstructed.cpu().detach()
encoded_cpu = encoded.cpu().detach()

K = 16
grid_cols = int(math.sqrt(K))
grid_rows = int(math.ceil(K / grid_cols))

for i in range(n_samples):
    fig = plt.figure(figsize=(grid_cols*2, grid_rows*2 + 4))  # extra height for original/recon

    # Original
    ax_orig = fig.add_subplot(grid_rows + 1, grid_cols, 1)
    ax_orig.imshow(images_cpu[i,0], cmap='gray', vmin=0, vmax=1)
    ax_orig.set_title("Original")
    ax_orig.axis('off')

    # Reconstructed
    ax_recon = fig.add_subplot(grid_rows + 1, grid_cols, 2)
    ax_recon.imshow(recon_cpu[i,0], cmap='gray', vmin=0, vmax=1)
    ax_recon.set_title("Reconstructed")
    ax_recon.axis('off')

    # Encoded channels (first K)
    for ch in range(K):
        ch_img = encoded_cpu[i, ch].numpy()
        ch_img -= ch_img.min()
        ch_img /= ch_img.max() + 1e-8

        row = 1 + (ch // grid_cols)
        col = ch % grid_cols
        ax = fig.add_subplot(grid_rows + 1, grid_cols, grid_cols * row + col + 1)
        ax.imshow(ch_img, cmap='gray')
        ax.set_title(f"ch {ch}", fontsize=6)
        ax.axis('off')

    plt.suptitle(f"Sample {i}")
    plt.tight_layout()
    plt.savefig('test_encoded.png')

#
"""
clustering
"""
#
print("Clustering started...")
if cluster_algo == 'kmeans':
    clustering = KMeans(n_clusters=cluster_no, random_state=0, n_init="auto").fit_predict(encoded_np)
elif cluster_algo == 'dbscan':
    clustering = HDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(encoded_np)
    print(clustering)
    cluster_no = np.max(clustering) + 1
elif cluster_algo == 'som':
    som = SOM(n=int(np.sqrt(cluster_no)), m=int(np.sqrt(cluster_no)), dim=encoded_np.shape[1])
    clustering = som.fit_predict(encoded_np, epochs=loops)
    som_transform = som.transform(encoded_np)  # Euclidean distance from each sample to each cluster centre
print("Clustering completed")

short_fname = []
for file_idx, file in enumerate(selected_files):
    name_from_file = extract_id_from_filename(file)
    short_fname.append(name_from_file)
name_to_cluster = dict(zip(short_fname, clustering))

best_clusters = []
sec_best_clusters = []
worst_clusters = []
for i, dists in enumerate(som_transform):
    idx = np.argpartition(dists, 2)[:2]
    best_cluster, sec_best_cluster = dists[idx]
    best_cluster, sec_best_cluster = np.sort([best_cluster, sec_best_cluster])
    best_clusters.append(best_cluster)
    sec_best_clusters.append(sec_best_cluster)
    worst_cluster = np.max(dists)
    worst_clusters.append(worst_cluster)
cluster_diff = np.array(worst_clusters) - np.array(best_clusters)
sil_diff = ( np.array(sec_best_clusters) - np.array(best_clusters) ) / np.array(sec_best_clusters)

sil_samp = silhouette_samples(encoded_np, clustering)

fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)
n_bins = 15
# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(sil_samp, bins=n_bins)
axs[0].set_title('Silhouette coefficient')
axs[1].hist(sil_diff, bins=n_bins)
axs[1].set_title('Best vs next best clustering')
plt.savefig('test_hist.png')

if add_flipped_images == False:
    sil_samp = sil_samp.reshape(-1)
    sil_diff = sil_diff.reshape(-1)
else:
    sil_samp = sil_samp.reshape(-1, 4)
    sil_diff = sil_diff.reshape(-1,4)


N = som_transform.shape[0]    # number of samples
K = som_transform.shape[1]    # number of SOM nodes
h = som.n
w = som.m

bmus = np.argmin(som_transform, axis=1)
coords = np.column_stack((bmus // w, bmus % w))   # (row, col)

# som map plot
map_image_save = f'{date}_{cluster_algo}_epochs{epochs}_{cluster_no}clusters_{times_noise}noise_som_map.png'
if cluster_algo == 'som':
    node_images = defaultdict(list)

    for idx, (r, c) in enumerate(coords):
        node_images[(r, c)].append(all_data[idx])
    
    """
    rep_images = {}

    for node, imgs in node_images.items():
        r, c = node
        node_index = r * w + c

    # distances for all samples assigned to this node
    dists = som_transform[bmus == node_index][:, node_index]
    
    best_idx = np.argmin(dists)

    # representative image is the closest sample
    rep_images[node] = imgs[best_idx]

    fig, axes = plt.subplots(som.n, som.m, figsize=(10,10))

    for r in range(som.n):
        for c in range(som.m):
            ax = axes[r, c]

            if (r, c) in rep_images:
                img = rep_images[(r, c)]
                if img.ndim == 3 and img.shape[0] in (1,3):  
                    img = img.transpose(1,2,0)   # CHW → HWC

                ax.imshow(img, cmap='plasma' if img.ndim==2 else None)
            else:
                ax.axis('off')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("som_representatives.png")  
    """

    hits = np.zeros((som.n, som.m))  # make grid the size of the som nodes

    for node_index in clustering:
        i, j = divmod(node_index, som.m)
        hits[i, j] += 1

    plt.figure()
    plt.imshow(hits, cmap="plasma", origin="lower")
    plt.colorbar(label="Number of samples")
    plt.title(f"SOM Map (n={int(np.sqrt(cluster_no))}, m={int(np.sqrt(cluster_no))}, epochs={loops})")
    plt.xlabel("SOM x-dimension")
    plt.ylabel("SOM y-dimension")
    plt.savefig(map_image_save)


sil_score = metrics.silhouette_score(encoded_np, clustering, metric='euclidean')
cal_h_score = metrics.calinski_harabasz_score(encoded_np, clustering)
dav_b_score = metrics.davies_bouldin_score(encoded_np, clustering)
print(f'Silhouette coefficient score: {sil_score}')     #higher is better
print(f'Calinski-Harabasz sndex score: {cal_h_score}')      #higher is better
print(f'Davies-Bouldin index score: {dav_b_score}')     #lower is better

#
"""
find cluster information
"""
#

# read in cluster information
number_labels = []
distances_og = []
classifications_og = []

cluster_info = open("table1.dat", "r")
for line in cluster_info:
    line_list = line.split()
    number_label = line_list[3]
    number_labels.append(number_label)
    distance = line_list[8]
    distances_og.append(distance)
    try:
        classification = line_list[17]
    except:
        classification = line_list[10]
    classifications_og.append(classification)
cluster_info.close()

# need to reorder to correspond to correct files
distances_ordered = []
classifications_ordered = []
unmatched = []
matched_files = []
matched_clustering = []

used_labels = set()

for file_idx, file in enumerate(selected_files):
    number_from_file = extract_id_from_filename(file)
    for idx, no_label in enumerate(number_labels):
        if number_from_file == no_label:
            matched_clustering.append(clustering[file_idx])
            if no_label not in used_labels:
                distances_ordered.append(distances_og[idx])
                classifications_ordered.append(classifications_og[idx])
                matched_files.append(file)
                used_labels.add(no_label)
            break

# ensure that the clustering numbers are integers not strings
matched_clustering = np.array(matched_clustering, dtype = int) 
if add_flipped_images == True:
    matched_clustering = matched_clustering.reshape(-1,4)

print(f"{len(selected_files) - len(matched_files)} images don't have corresponding info")

expected_prop = {
    "Q": 120/1002,
    "P": 143/1002,
    "Y": 275/1002,
    "H": 219/1002,
    "na": 245/1002
}

with open(classification_result_save, "w") as f:

    class_and_cluster = np.column_stack((classifications_ordered, matched_clustering))
    classes = ["Q", "P", "Y", "H", "na"]
    tally = np.zeros(len(classes))
    for x in range(cluster_no):              # loop through each cluster
        tally = np.zeros(len(classes))       # clear the tally for each cluster
        for row in class_and_cluster:        # loop through each row of the clusters
            for cluster_val in row[1:]:
                if int(cluster_val) == x:    # if the cluster matches the current cluster
                    for idx, letter in enumerate(classes):      # loop through each class label
                        if row[0] == letter:                    # if the class of the current row matches the current letter
                            tally[idx] += 1                     # add one to the tally for this class
        clust_size = tally.sum()
        exp = np.array([expected_prop[c] * clust_size for c in classes])
        obs = tally
        chi2_stat, p_value = chisquare(f_obs=obs, f_exp=exp)
        counts_str = " ".join(f"{letter}:{int(tally[idx])}" for idx, letter in enumerate(classes))
        #print(f"cluster {x}: {counts_str}")
        #print("Expected:", np.round(exp, 2))
        #print(f"Chi-square: {chi2_stat:.3f}, p-value: {p_value:.4f}")
        out = (
            f"cluster {x}: {counts_str}\n"
            f"Expected: {np.round(exp, 2)}\n"
            f"Chi-square: {chi2_stat:.3f}, p-value: {p_value:.4f}\n\n")
        f.write(out) # also write to file

# ------------------------------------------------------------------
# Compute per-node class proportions
# ------------------------------------------------------------------
classes = ["Q", "P", "Y", "H", "na"]
K = som.n * som.m

# Map each BMU index to (row, col)
bmus = clustering
node_rows = bmus // som.m
node_cols = bmus % som.m

# Hold counts and proportions
node_class_counts = {k: np.zeros(len(classes), dtype=float) for k in range(K)}

# Count classes per node
for idx, node_idx in enumerate(bmus):
    cls = classifications_ordered[idx]
    if cls in classes:
        ci = classes.index(cls)
        node_class_counts[node_idx][ci] += 1

# Convert counts → proportions per node
node_class_prop = {}
for k in range(K):
    total = np.sum(node_class_counts[k])
    if total > 0:
        node_class_prop[k] = node_class_counts[k] / total
    else:
        node_class_prop[k] = np.zeros(len(classes))

def make_dictionary():
    df_switch = pd.read_csv("/raid/scratch/mevans/table1.dat", delimiter=' ', header=None, usecols=[1,3],
                    dtype={1: str, 3: str}, skipinitialspace=True)
    bad_names = df_switch[1]
    good_names = df_switch[3]
    convert_name = dict(zip(good_names, bad_names))
    return convert_name

def file_switch(dictionary, good_file):
    bad_file = dictionary[good_file]
    return bad_file

def read_core_info(selected_files):
    convert_name = make_dictionary()
    df = pd.read_csv("/raid/scratch/hannahb/table3.dat", delimiter=' ', header=0, usecols=["Name","Fpeak"],
                    dtype={"Name": str, "Fpeak": np.float64}, skipinitialspace=True)
    #names = df['Name']
    #peak_fluxes = df['Fpeak']
    names = df['Name'].tolist()       # convert to list
    peak_fluxes = df['Fpeak'].tolist()
    source_names = []
    number_of_cores = []
    max_fluxes = []
    core_lines = []
    for file_idx, file in enumerate(selected_files):
        name_from_file = extract_id_from_filename(file)
        try: 
            bad_name = file_switch(convert_name, name_from_file)
        except:
            bad_name = None
            pass
        if bad_name is not None and bad_name in names:
            count = names.count(bad_name)
            number_of_cores.append(count)
            source_names.append(name_from_file)
            indices = [i for i, n in enumerate(names) if n == bad_name]
            fluxes = [peak_fluxes[i] for i in indices]
            max_flux = max(fluxes) if fluxes else 0.0
            max_fluxes.append(max_flux)
            core_lines.append(f"{name_from_file}, {count}, {max_flux}")
        else:
            number_of_cores.append(0)
            source_names.append(name_from_file)
            max_fluxes.append(0)
    return source_names, number_of_cores, max_fluxes, core_lines

sn, noc, mf, cl = read_core_info(selected_files)
with open(core_test_results_save, "w") as f:
    f.write("\n".join(cl))

core_clusters = []
for idx, name in enumerate(sn):
    core_clusters.append(name_to_cluster[name])

def make_cluster_tally(noc, clusters):
    cluster_no=16
    cluster_list = list(range(0,cluster_no+1))
    class_and_cluster = np.column_stack((noc, clusters))
    noc_options = list(range(0,np.max(noc)+1))
    tally = np.zeros(len(noc_options))
    combined_tally = []
    for x in range(cluster_no):              # loop through each cluster
        tally = np.zeros(len(noc_options))       # clear the tally for each cluster
        for row in class_and_cluster:        # loop through each row of the clusters
            for cluster_val in row[1:]:
                if int(cluster_val) == x:    # if the cluster matches the current cluster
                    for idx, numb in enumerate(noc_options):      # loop through each class label
                        if row[0] == numb:                    # if the class of the current row matches the current letter
                            tally[idx] += int(1)                     # add one to the tally for this class
        clust_size = tally.sum()
        combined_tally.append(tally)
    tally_dict = dict(zip(cluster_list, combined_tally))
    return tally_dict, noc_options

def core_bar_chart(results, category_names):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['Spectral'](
        np.linspace(0.15, 0.85, data.shape[1])
    )

    fig, ax = plt.subplots(figsize=(9.2, 6))
    ax.invert_yaxis()
    ax.xaxis.set_visible(True)
    ax.set_xlabel("Number of images")   # optional label
    ax.set_xlim(0, np.sum(data, axis=1).max())


    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)

    # Adjust legend to stay inside figure
    ax.legend(ncols=4, bbox_to_anchor=(1.02, 1),
          loc='upper left', fontsize='small', frameon=False)        

    plt.tight_layout()
    return fig, ax

tally_dict, noc_options = make_cluster_tally(noc, core_clusters)
core_bar_chart(tally_dict, noc_options)
plt.savefig('core_number_stack_test.png')

def neuron_class_pie_charts(node_class_prop, classes):
    labels = list(node_class_prop.keys())
    data = np.array([node_class_prop[k] for k in labels])

    n_clusters = len(labels)
    n_cols = int(math.ceil(math.sqrt(n_clusters)))
    n_rows = int(math.ceil(math.sqrt(n_clusters)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    category_colors = plt.colormaps['rainbow'](
        np.linspace(0.15, 0.85, len(classes))
    )

    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]

            if idx < n_clusters:
                props = data[idx]
                if np.sum(props) == 0:
                    ax.axis("off")
                    idx += 1
                    continue

                ax.pie(
                    props,
                    colors=category_colors
                )
                ax.set_title(f"Neuron {labels[idx]}", fontsize=14)

            else:
                ax.axis("off")

            idx += 1

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=category_colors[i], label=classes[i])
                      for i in range(len(classes))]

    fig.legend(
        handles=legend_handles,
        title="Classes",
        loc='lower center',
        ncol=len(classes),
        frameon=False,
        fontsize=18,
        title_fontsize=18,
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=fig.transFigure
    )

    return fig, axes


    from matplotlib.patches import Patch

    max_cols = 10  # max number of legend entries per row
    ncol = min(len(category_names), max_cols)
    nrows = int(np.ceil(len(category_names) / max_cols))

    legend_handles = [Patch(facecolor=category_colors[i], label=str(category_names[i]))
                    for i in range(len(category_names))]

    plt.subplots_adjust(bottom=0.18)  # leave more space for the legend

    fig.legend(
        handles=legend_handles,
        title="Number of cores:",
        loc='lower center',
        ncol=ncol,
        frameon=False,
        fontsize=20,          # increase font size
        title_fontsize=20,    # increase title size
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=fig.transFigure
        )

    #plt.tight_layout()
    return fig, axes

fig, axes = neuron_class_pie_charts(node_class_prop, classes)
plt.savefig("neuron_class_piecharts.png")


#
"""
save results
"""
#

results_array = np.column_stack((matched_files, classifications_ordered, matched_clustering))

# find instances of each cluster
for i in range(cluster_no):
    count = np.count_nonzero(clustering == i)
    print(f"Images clustered in {i}: {count}")

def extract_number(path):
    # Get only the filename
    fname = str(path).split("/")[-1]
    # Extract the first continuous number
    match = re.search(r"\d+", fname)
    return match.group(0) if match else fname

lines = []
for i, file in enumerate(matched_files):
    clas = classifications_ordered[i]
    clusters = matched_clustering[i]  
    silco_samp = sil_samp[i]
    sil_dif = sil_diff[i]
    clusters = np.atleast_1d(clusters)
    silco_samp = np.atleast_1d(silco_samp)
    sil_dif = np.atleast_1d(sil_dif)
    clusters_str = "[" + ", ".join(map(str, clusters)) + "]"
    counts = Counter(clusters)
    counts_str = ", ".join(f"{val}:{count}" for val, count in counts.items())
    short_name = extract_number(file)
    lines.append(f"{short_name}, {clas}\t{clusters_str}\t{counts_str}\t{silco_samp}\t{sil_dif}")

with open(text_results_save, "w") as f:
    f.write("\n".join(lines))

#
"""
view clustered images
"""
#
os.makedirs(cluster_examples_dir, exist_ok=True)

unique_clusters = np.unique(clustering)

cluster_to_files = {i: [] for i in unique_clusters}

for idx, cluster_id in enumerate(clustering):
    fname = selected_files[idx]
    if fname not in cluster_to_files[cluster_id]:
        cluster_to_files[cluster_id].append(fname)

# Helper: filename → index in dataset
filename_to_index = {f: i for i, f in enumerate(selected_files)}

# Process each cluster
for cluster_id, filenames in cluster_to_files.items():

    if len(filenames) == 0:
        print(f"Cluster {cluster_id} has no images, skipping.")
        continue

    # pick up to grid_size² unique filenames
    chosen_files = filenames[:grid_size**2]

    # prepare two figures
    fig1 = plt.figure(figsize=(grid_size*2, grid_size*2))  # normalized
    fig2 = plt.figure(figsize=(grid_size*2, grid_size*2))  # processed

    for ax_idx, fname in enumerate(chosen_files):
        # get dataset index
        sample_idx = filename_to_index[fname]

        # extract images
        normalised_image = normalised_datas[sample_idx]
        processed_image = new_datas[sample_idx]
        base = os.path.basename(fname)

        # subplots
        ax1 = fig1.add_subplot(grid_size, grid_size, ax_idx + 1)
        ax2 = fig2.add_subplot(grid_size, grid_size, ax_idx + 1)

        # plot normalized
        ax1.imshow(normalised_image, vmin=np.min(normalised_image), vmax=np.max(normalised_image), origin='lower')
        ax1.set_title(base, fontsize=6)
        ax1.axis('off')

        # plot processed
        ax2.imshow(processed_image, origin='lower', vmin=np.min(processed_image), vmax=np.max(processed_image))
        ax2.set_title(base, fontsize=6)
        ax2.axis('off')

    # Save normalized grid
    fig1.suptitle(f"Cluster {cluster_id}: Normalised")
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    out1 = os.path.join(cluster_examples_dir, f"cluster_{cluster_id:03d}_normalized.png")
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)

    # Save processed grid
    fig2.suptitle(f"Cluster {cluster_id}: Processed")
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    out2 = os.path.join(cluster_examples_dir, f"cluster_{cluster_id:03d}_processed.png")
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)

  
#
"""
plot clustering map
"""
#
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

plot_cluster_map = False
if plot_cluster_map == True:
    # --- Run t-SNE or PCA ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    encoded_2d = tsne.fit_transform(encoded_np)

    # --- Normalize image data for display ---
    images = datas.squeeze()  # remove channel dim if (N,1,H,W)
    images = (images - images.min()) / (images.max() - images.min())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1],
                        c=clustering, cmap='tab20', s=15, alpha=0.8)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title("Clusters in latent space (dots)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig("test_latent_clusters_dots.png", dpi=200)

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(12, 10))

    # Optional: background scatter (for density)
    scatter = ax.scatter(encoded_2d[:, 0], encoded_2d[:, 1], 
                        c=clustering, cmap='tab10', s=10, alpha=0.3)

    # --- Overlay image thumbnails ---
    def imscatter(x, y, images, ax=None, zoom=0.08):
        if ax is None:
            ax = plt.gca()
        for (x0, y0, img) in zip(x, y, images):
            imagebox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imagebox, (x0, y0), frameon=False)
            ax.add_artist(ab)
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()

    # You can subsample if too many images (for speed)
    max_thumbs = 1006
    idxs = np.random.choice(len(images), min(len(images), max_thumbs), replace=False)

    imscatter(encoded_2d[idxs, 0], encoded_2d[idxs, 1], images[idxs], ax=ax, zoom=0.08)

    plt.title("t-SNE with image thumbnails", fontsize=14)
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.savefig('test_map2.png')
