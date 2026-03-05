from glob import glob
import json
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from time import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib
from tqdm import tqdm
import sys
from functools import lru_cache
from omegaconf import DictConfig, OmegaConf
import hydra

def load_feat(paths, max_num=24770, max_memory=10, max_sec=3600):
    num = 0
    feats = []
    ts = time()
    for path in paths:
        feat = np.load(path)
        if len(feats) > 0:
            t = time() - ts
            out = np.concatenate(feats)
            memory = sys.getsizeof(out)*(10**-9)
            if memory > max_memory:
                msg = "Memory out"
                print("%d samples, %.3f GB, %.1f sec ellapsed"%(num, memory, t))
                return out, num, memory, t, msg
            if t >= max_sec:
                msg = "Time out"
                print("%d samples, %.3f GB, %.1f sec ellapsed"%(num, memory, t))
                return out, num, memory, t, msg
            if num >= max_num:
                msg = "Num out"
                print("%d samples, %.3f GB, %.1f sec ellapsed"%(num, memory, t))
                return out, num, memory, t, msg
        feats.append(feat)
        num += 1
    t = time() - ts
    out = np.concatenate(feats)
    memory = sys.getsizeof(out)*(10**-9)
    msg = "Conditions required were completed"
    print("%d samples, %.3f GB, %.1f sec ellapsed"%(num, memory, t))
    return out, num, memory, t, msg

def get_kmeans_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
    random_state,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        tol=tol,
        max_no_improvement=max_no_improvement,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
        random_state=random_state,
        verbose=1,
        compute_labels=True,
        init_size=None,
    )

def train_kmeans(kmeans_model, features_batch):
    start_time = time()
    kmeans_model.fit(features_batch)
    time_taken = round((time() - start_time) // 60, 2)
    return kmeans_model, time_taken


@hydra.main(version_base=None, config_path="../configs", config_name="feat2unit")
def main(cfg:DictConfig):
    model_name = cfg.km_name
    input_feature_dir = os.path.join(cfg.data_path, 'preprocessed', 'target_feature', cfg.feature.target, f'{cfg.feature.sub_option}{cfg["feature"][cfg.feature.sub_option]}')
    out_kmeans_model_path = os.path.join(input_feature_dir.replace('target_feature', 'target_unit'), f'{model_name}.bin')
    if os.path.isfile(out_kmeans_model_path):
        ans = None
        while ans != 'y' and ans != 'n':
            ans = input(f"There exist {out_kmeans_model_path} aleady. Do you want to overwrite?(y/n) ")
        if ans == 'n':
            sys.exit()

    input_feature_path = glob(os.path.join(input_feature_dir, 'train/nonparallel_data/*/*.npy'))
    input_feature_path += glob(os.path.join(input_feature_dir, 'train/voiced_parallel_data/*/*.npy'))
    features_batch, num, memory, t, msg = load_feat(input_feature_path, cfg.max_num, cfg.max_memory, cfg.max_sec)
    print(f"Features shape = {features_batch.shape}\n")
    # Learn and save K-means model
    kmeans_model = get_kmeans_model(
        n_clusters=cfg.num_clusters,
        init=cfg.init,
        max_iter=cfg.max_iter,
        batch_size=cfg.batch_size,
        tol=cfg.tol,
        max_no_improvement=cfg.max_no_improvement,
        n_init=cfg.n_init,
        reassignment_ratio=cfg.reassignment_ratio,
        random_state=cfg.seed,
    )
    print("Starting k-means training...")
    kmeans_model, time_taken = train_kmeans(
        kmeans_model=kmeans_model, features_batch=features_batch
    )
    print(f"...done k-means training in {time_taken} minutes")
    inertia = -kmeans_model.score(features_batch) / len(features_batch)
    print(f"Total intertia: {round(inertia, 2)}\n")

    print(f"Saving k-means model to {out_kmeans_model_path}")
    os.makedirs(os.path.dirname(out_kmeans_model_path), exist_ok=True)
    joblib.dump(kmeans_model, open(out_kmeans_model_path, "wb"))

    # Validation
    valid_feature_path = glob(os.path.join(input_feature_dir, 'dev/nonparallel_data/*/*.npy'))
    valid_feature_path += glob(os.path.join(input_feature_dir, 'dev/voiced_parallel_data/*/*.npy'))
    features_batch, _, _, _, _ = load_feat(valid_feature_path, cfg.max_num, cfg.max_memory, cfg.max_sec)
    inertia_dev = -kmeans_model.score(features_batch) / len(features_batch)
    print(f"Valid intertia: {round(inertia_dev, 2)}\n")


    print("%d samples, %.3f GB, %.1f sec ellapsed"%(num, memory, t))
    print(msg)



if __name__ == "__main__":
    main()
