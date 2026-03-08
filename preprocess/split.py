import os
import glob
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict
from omegaconf import OmegaConf


def _load_dataset_dir_from_common() -> str:
    common_yaml = Path(__file__).resolve().parents[1] / "configs" / "common.yaml"
    default_dataset_dir = "/data/path/silent_speech_dataset"
    if not common_yaml.exists():
        return default_dataset_dir
    cfg = OmegaConf.load(common_yaml)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        return default_dataset_dir
    paths = resolved.get("paths", {})
    if not isinstance(paths, dict):
        return default_dataset_dir
    return str(paths.get("silent_speech_dataset", default_dataset_dir))


DATASET_DIR = _load_dataset_dir_from_common()
BASE_DIR = os.path.dirname(DATASET_DIR)


def collect_files(data_type: str):
    pattern = os.path.join(DATASET_DIR, data_type, "*", "data", "emg", "emg*.npz")
    file_paths = glob.glob(pattern, recursive=True)

    result = defaultdict(list)
    for path in file_paths:

        rel_from_dataset = os.path.relpath(path, DATASET_DIR)
        parts = rel_from_dataset.split(os.sep)

        # parts: [data_type, session, "data", "emg", filename]
        if len(parts) < 5:
            continue

        session = parts[1]
        filename = parts[-1]  # emg*{data_num}*.npz



        import re

        nums = re.findall(r"\d+", filename.split('.')[0].split('_')[-1])
        if not nums:
            continue
        data_num = nums[-1]

        key = (session, data_num)
        result[key].append(path)

    return result


def split_by_session_keys(keys, split_ratio, seed):
    """
    keys: [(session, data_num), ...]
    """
    random.seed(seed)

    session_to_keys = defaultdict(list)
    for session, data_num in keys:
        session_to_keys[session].append((session, data_num))

    split_assign = {}  # (session, data_num) -> split

    for session, sess_keys in session_to_keys.items():

        random.shuffle(sess_keys)

        n = len(sess_keys)
        train_ratio, valid_ratio, test_ratio = split_ratio

        n_train = int(round(n * train_ratio))
        n_valid = int(round(n * valid_ratio))
        if n_train + n_valid > n:

            overflow = n_train + n_valid - n
            n_valid = max(0, n_valid - overflow)

        n_test = n - n_train - n_valid

        train_keys = sess_keys[:n_train]
        valid_keys = sess_keys[n_train:n_train + n_valid]
        test_keys = sess_keys[n_train + n_valid:]

        for k in train_keys:
            split_assign[k] = "train"
        for k in valid_keys:
            split_assign[k] = "valid"
        for k in test_keys:
            split_assign[k] = "test"

    return split_assign


def build_split_csv(
    split_ratio=(0.8, 0.1, 0.1),
    seed: int = 42,
    output_csv: str = "emg_split.csv",
):
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "sum of split_ratio should be 1."


    silent_map = collect_files("silent")
    voiced_map = collect_files("voiced")


    silent_keys = list(silent_map.keys())


    split_assign = split_by_session_keys(silent_keys, split_ratio, seed)

    rows = []


    for key, split in split_assign.items():
        session, data_num = key
        paths = silent_map.get(key, [])
        if not paths:
            continue
        for full_path in paths:
            rel_path = os.path.relpath(full_path, BASE_DIR)
            rows.append(
                {
                    "session": session,
                    "data_type": "silent",
                    "data_num": data_num,
                    "path": rel_path,
                    "split": split,
                }
            )


    for key, split in split_assign.items():
        session, data_num = key
        v_paths = voiced_map.get(key)
        if not v_paths:
            print(f"[WARNING] voiced file not found: session={session}, data_num={data_num}")
            continue
        for full_path in v_paths:
            rel_path = os.path.relpath(full_path, BASE_DIR)
            rows.append(
                {
                    "session": session,
                    "data_type": "voiced",
                    "data_num": data_num,
                    "path": rel_path,
                    "split": split,
                }
            )


    df = pd.DataFrame(rows, columns=["session", "data_type", "data_num", "path", "split"])
    df.to_csv(output_csv, index=False, encoding="utf-8")



    session_split_counts = (
        df.groupby(["session", "split"])["path"].count().reset_index(name="count")
    )

    session_total_counts = df.groupby("session")["path"].count().reset_index(name="total")

    split_total_counts = df.groupby("split")["path"].count().reset_index(name="total")

    print("\n[split count]")
    for _, row in session_split_counts.iterrows():
        print(f"session={row['session']}, split={row['split']}, count={row['count']}")

    print("\n[session count]")
    for _, row in session_total_counts.iterrows():
        print(f"session={row['session']}, total={row['total']}")

    print("\n[final split count]")
    for _, row in split_total_counts.iterrows():
        print(f"split={row['split']}, total={row['total']}")

    return df


if __name__ == "__main__":


    build_split_csv(split_ratio=(0.85, 0.075, 0.075), seed=42, output_csv="emg_split.csv")

