from glob import glob
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import os
from torch.utils.data import DataLoader
import joblib
import sys
from time import time

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from utils import merge

@hydra.main(version_base=None, config_path="../configs", config_name="feat2unit")
def main(cfg:DictConfig):
    sub_option = f'{cfg.feature.sub_option}{cfg["feature"][cfg.feature.sub_option]}'
    input_feature_dir = os.path.join(cfg.data_path, 'preprocessed', f'{cfg.target}_feature', cfg.feature.target, sub_option)
    if cfg.target == 'est':
        input_feature_dir = os.path.join(input_feature_dir, cfg.encoder)
    input_feature_path = glob(os.path.join(input_feature_dir, '*/nonparallel_data/*/*.npy'))
    input_feature_path += glob(os.path.join(input_feature_dir, '*/voiced_parallel_data/*/*.npy'))
    if cfg.target == 'est':
        input_feature_path += glob(os.path.join(input_feature_dir, '*/silent_parallel_data/*/*.npy'))
    print(len(input_feature_path))
    if len(input_feature_path) < 1:
        print(f'{input_feature_dir} is not valid')
        sys.exit()

    output_unit_dir = os.path.join(input_feature_dir.replace(f'{cfg.target}_feature', f'{cfg.target}_unit'), f'{cfg.km_name}')
    kmeans_model_path = os.path.join(cfg.data_path, 'preprocessed/target_unit', cfg.feature.target, sub_option, f'{cfg.km_name}.bin')

    print(f"Loading K-means model from {kmeans_model_path} ...")
    km = joblib.load(open(kmeans_model_path, 'rb'))
    km.verbose = False

    ts = time()
    for path in input_feature_path:
        feat = np.load(path)
        pred = km.predict(feat) + 1
        pred_str = " ".join(str(p) for p in pred)
        output_path = path.replace(input_feature_dir, output_unit_dir)
        output_path = output_path.replace('.npy', '.txt')
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as fout:
            fout.write(pred_str)

    time_taken = round(time() - ts, 2)
    print(f"{time_taken:.1f} sec ellapsed")

if __name__ == "__main__":
    main()

