import difflib
import hydra
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import os
import sys
from textgrids import TextGrid
import torch

def load_partial_pretrained_model(model, ckpt_path, selected_channels):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # conv1: manual slicing
    conv1_key = "conv_blocks.0.conv1.weight"
    if conv1_key in state_dict:
        pretrained_conv1_weight = state_dict[conv1_key]  # (out_ch, 8, kernel)
        new_conv1_weight = pretrained_conv1_weight[:, selected_channels, :]
        model.conv_blocks[0].conv1.weight.data.copy_(new_conv1_weight)
        print(f"[Info] Adapted conv1.weight to selected channels: {selected_channels}")

    # residual_path.weight: same slicing as conv1
    residual_key = "conv_blocks.0.residual_path.weight"
    if residual_key in state_dict:
        pretrained_residual_weight = state_dict[residual_key]  # (out_ch, 8, 1)
        new_residual_weight = pretrained_residual_weight[:, selected_channels, :]
        model.conv_blocks[0].residual_path.weight.data.copy_(new_residual_weight)

    # residual_path.bias
    residual_bias_key = "conv_blocks.0.residual_path.bias"
    if residual_bias_key in state_dict:
        model.conv_blocks[0].residual_path.bias.data.copy_(state_dict[residual_bias_key])

    # res_norm (BN): gamma, beta
    for bn_param in ["weight", "bias", "running_mean", "running_var"]:
        key = f"conv_blocks.0.res_norm.{bn_param}"
        if key in state_dict:
            getattr(model.conv_blocks[0].res_norm, bn_param).data.copy_(state_dict[key])

    # load remaining parts of the model (skip those handled above)
    skip_keys = {
        "conv_blocks.0.conv1.weight",
        "conv_blocks.0.residual_path.weight",
        "conv_blocks.0.residual_path.bias",
        "conv_blocks.0.res_norm.weight",
        "conv_blocks.0.res_norm.bias",
        "conv_blocks.0.res_norm.running_mean",
        "conv_blocks.0.res_norm.running_var",
    }
    filtered_state_dict = {k: v for k, v in state_dict.items() if k not in skip_keys}
    
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    print(f"[Info] Loaded partial weights. Missing keys: {missing}")
    print(f"[Info] Loaded partial weights. Unexpected keys: {unexpected}")
    return model

def edit_distance_sequences(x, y):
    N1, N2 = len(x), len(y)
    dp = [[0] * (N2 + 1) for _ in range(N1 + 1)]

    # Initialize the dp table
    for i in range(N1 + 1):
        dp[i][0] = i  # Cost of deleting all characters from x to match empty y
    for j in range(N2 + 1):
        dp[0][j] = j  # Cost of inserting all characters of y into empty x

    # Fill dp table
    for i in range(1, N1 + 1):
        for j in range(1, N2 + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + 1  # Substitution
                )

    # Backtracking to find edit_x and edit_y
    edit_x = [0] * N1
    edit_y = [0] * N2

    i, j = N1, N2
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            edit_x[i - 1] = 0
            edit_y[j - 1] = 0
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            edit_x[i - 1] = 1
            edit_y[j - 1] = 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            edit_x[i - 1] = 2
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            edit_y[j - 1] = 3
            j -= 1

    while i > 0:
        edit_x[i - 1] = 2
        i -= 1

    while j > 0:
        edit_y[j - 1] = 3
        j -= 1

    return edit_x, edit_y

def init_cfg(config_name: str):
    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(config_name=config_name)
    return cfg

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def extract_path_info(path):
    base_dir, file_name = os.path.split(path)
    idx = file_name.split('_')[0]
    parts = base_dir.split('/')
    data_split = parts[-3]
    data_type = parts[-2]
    sess = parts[-1]
    return data_split, data_type, sess, idx


def extract_phoneme(path):
    tg = TextGrid(path)
    phones = []
    for interval in tg['phones']:
        phone = interval.text.lower()
        phones.append(phone)
    return phones

if __name__ == '__main__':
    path = '/data2/spjune/silent_speech/silent_speech_dataset/train/nonparallel_data/4-29/112_tg.TextGrid'
    print(extract_phoneme(path))

@jit(nopython=True)
def time_warp(costs):
    dtw = np.zeros_like(costs)
    dtw[0,1:] = np.inf
    dtw[1:,0] = np.inf
    eps = 1e-4
    for i in range(1,costs.shape[0]):
        for j in range(1,costs.shape[1]):
            dtw[i,j] = costs[i,j] + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1,j-1])
    return dtw

def align_from_distances(distance_matrix, debug=False):
    # for each position in spectrum 1, returns best match position in spectrum2
    # using monotonic alignment
    dtw = time_warp(distance_matrix)

    i = distance_matrix.shape[0]-1
    j = distance_matrix.shape[1]-1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min([(i-1,j),(i,j-1),(i-1,j-1)], key=lambda x: dtw[x[0],x[1]])

    if debug:
        visual = np.zeros_like(dtw)
        visual[range(len(results)),results] = 1
        plt.matshow(visual)
        plt.show()

    return results
