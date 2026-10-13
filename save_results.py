import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import re
from collections import Counter
import math

text_results_save = f'COMBINEDconvae_result.txt'
core_test_results_save = f'COMBINEDcore_info.txt'
classification_result_save = f'COMBINEDclassification_info.txt'

def define_save_paths(config):
    text_results_path = os.path.join(config.run_dir, text_results_save)
    core_test_results_path = os.path.join(config.run_dir, core_test_results_save)
    classification_results_path = os.path.join(config.run_dir, classification_result_save)
    return text_results_path, core_test_results_path, classification_results_path


def make_dictionary():
    df_switch = pd.read_csv("/raid/scratch/hannahb/table1.dat", delimiter=' ', header=None, usecols=[1,3],
                    dtype={1: str, 3: str}, skipinitialspace=True)
    bad_names = df_switch[1]
    good_names = df_switch[3]
    convert_name = dict(zip(good_names, bad_names))
    return convert_name

def file_switch(dictionary, good_file):
    bad_file = dictionary[good_file]
    return bad_file

def read_cluster_info():
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
    return number_labels, distances_og, classifications_og

def extract_id_from_filename(filepath):
    # get only the filename
    fname = os.path.basename(filepath)
    
    # match everything before "_cont"
    match = re.match(r'(.+?)_cont', fname)
    if match:
        return match.group(1)

def order_values(selected_files, number_labels, distances_og, classifications_og, clustering, config):
    distances_ordered = []
    classifications_ordered = []
    unmatched = []
    matched_files = []
    matched_clustering = []

    for file_idx, file in enumerate(selected_files):
        number_from_file = extract_id_from_filename(file)
        found_match = False
        for idx, no_label in enumerate(number_labels):
            if number_from_file == no_label:
                matched_clustering.append(clustering[file_idx])
                distances_ordered.append(distances_og[idx])
                classifications_ordered.append(classifications_og[idx])
                matched_files.append(file)
                found_match = True
                break
        if not found_match:
            unmatched.append(file)

    # ensure that the clustering numbers are integers not strings
    matched_clustering = np.array(matched_clustering, dtype = int) 

    print(f"{len(unmatched)} images don't have corresponding info")
    return classifications_ordered, matched_clustering, matched_files

def save_classification_info(classifications_ordered, matched_clustering, classification_results_path, config):
    write_classification_info(
        classifications_ordered,
        matched_clustering,
        classification_results_path,
        cluster_no=config.n_num * config.m_num,
        config=config,
    )


def write_classification_info(classifications_ordered, matched_clustering, classification_results_path, cluster_no, config=None):
    """Write cluster-wise class counts into COMBINEDclassification_info.txt.

    This is the shared writer for both SOM and GSOM runs.
    """
    expected_prop = {
        "Q": 120 / 1002,
        "P": 143 / 1002,
        "Y": 275 / 1002,
        "H": 219 / 1002,
        "na": 245 / 1002,
    }
    classes = ["Q", "P", "Y", "H", "na"]

    class_and_cluster = np.column_stack((classifications_ordered, matched_clustering))

    with open(classification_results_path, "w") as f:
        for x in range(cluster_no):
            tally = np.zeros(len(classes))
            for row in class_and_cluster:
                for cluster_val in row[1:]:
                    if int(cluster_val) == x:
                        for idx, letter in enumerate(classes):
                            if row[0] == letter:
                                tally[idx] += 1
            clust_size = tally.sum()
            exp = np.array([expected_prop[c] * clust_size for c in classes])
            obs = tally
            counts_str = " ".join(f"{letter}:{int(tally[idx])}" for idx, letter in enumerate(classes))
            out = (
                f"cluster {x}: {counts_str}\n"
                f"Expected: {np.round(exp, 2)}\n\n"
            )
            f.write(out)

    return

def extract_number(path):
        # Get only the filename
        fname = str(path).split("/")[-1]
        # Extract the first continuous number
        match = re.search(r"\d+", fname)
        return match.group(0) if match else fname

def plot_single_cluster(obs, exp, config, classes=None, cluster_idx=0):
    save_direc = os.path.join(config.run_dir, 'test_chi_sq_hist')
    os.makedirs(save_direc, exist_ok=True)
    if classes is None:
        classes = ['Q', 'P', 'Y', 'H', 'na']
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x - width/2, obs, width, label='Observed', color='skyblue')
    ax.bar(x + width/2, exp, width, label='Expected', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel('Counts')
    ax.set_title(f'Cluster {cluster_idx}: Observed vs Expected Counts')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if config.run_dir:
        plt.savefig(os.path.join(save_direc, f"cluster_{cluster_idx}_counts.png"))
        plt.close(fig)
    else:
        plt.show()

def save_clustering_results(matched_files, classifications_ordered, matched_clustering, clustering, sil_diff, sil_samp, config):
    cluster_no = config.n_num * config.m_num
    results_array = np.column_stack((matched_files, classifications_ordered, matched_clustering))

    # find instances of each cluster
    for i in range(cluster_no):
        count = np.count_nonzero(clustering == i)
        print(f"Images clustered in {i}: {count}")

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
    return

def save_all_results(selected_files, clustering, sil_diff, sil_samp, config):
    #global text_results_save, core_test_results_save, classification_result_save
    #text_results_save = os.path.join(config.run_dir, "COMBINEDconvae_result.txt")
    #core_test_results_save = os.path.join(config.run_dir, "COMBINEDcore_info.txt")
    #classification_result_save = os.path.join(config.run_dir, "COMBINEDclassification_info.txt")

    #text_results_save = f'convae_result.txt'
    #core_test_results_save = f'core_info.txt'
    #classification_result_save = f'classification_info.txt'


    text_results_path, core_test_results_path, classification_results_path = define_save_paths(config)
    number_labels, distances_og, classifications_og = read_cluster_info()
    classifications_ordered, matched_clustering, matched_files = order_values(
        selected_files, number_labels, distances_og, classifications_og, clustering, config)

    save_classification_info(classifications_ordered, matched_clustering, classification_results_path, config)
    save_clustering_results(matched_files, classifications_ordered, matched_clustering, clustering, sil_diff, sil_samp, config)
    return

    

