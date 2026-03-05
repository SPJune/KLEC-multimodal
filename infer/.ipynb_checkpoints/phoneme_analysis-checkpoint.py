import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import csv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from modules import EMGEncoder
from loader import EMGDataset
from loss import dtw_loss
from data_utils import phoneme_inventory

def save_phoneme_analysis_csv(confusion, acc_per_phoneme, csv_path='phoneme_analysis.csv'):
    phonemes = phoneme_inventory
    phoneme_totals = confusion.sum(axis=0)
    misclassified_counts = confusion.sum(axis=0) - np.diag(confusion)

    rows = []

    # Pairwise errors
    for i in range(len(phonemes)):
        for j in range(len(phonemes)):
            total = phoneme_totals[j]
            wrong = confusion[i, j]
            if i == j or total == 0 or wrong == 0:
                continue
            error = wrong / total
            rows.append({
                "type": "pair",
                "from": phonemes[i],
                "to": phonemes[j],
                "error": f"{error*100:.2f}",
                "wrong": int(wrong),
                "total": int(total)
            })

    # Per-phoneme errors
    for i in range(len(phonemes)):
        total = phoneme_totals[i]
        wrong = misclassified_counts[i]
        if total == 0:
            continue
        error = wrong / total
        rows.append({
            "type": "phoneme",
            "from": phonemes[i],
            "to": phonemes[i],
            "error": f"{error*100:.2f}",
            "wrong": int(wrong),
            "total": int(total)
        })

    # Write to CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["type", "from", "to", "error", "wrong", "total"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n[Saved CSV] {csv_path}")

def cal_ph_acc(model, dataloader, device):
    confusion = np.zeros((len(phoneme_inventory), len(phoneme_inventory)))
    total_correct = 0
    total_length = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating phoneme accuracy"):
            emg = batch['emg'].to(device)
            y = batch['speech_features'].to(device)
            y_ph = batch['phonemes'].to(device)
            silents = batch['silents'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            est_lengths = batch['est_lengths'].to(device)
            f0 = batch['f0'].to(device)
            uv = batch['uv'].to(device)

            pred_feat, pred_ph, _, _ = model(emg)

            _, _, _, acc = dtw_loss(pred_feat, pred_ph, y, y_ph, silents,
                                    model.hparams.phoneme_loss_weight,
                                    target_lengths, est_lengths,
                                    phoneme_eval=True,
                                    phoneme_confusion=confusion)

            total_correct += acc * y_ph.numel()
            total_length += y_ph.numel()

    acc_per_phoneme = np.zeros(len(phoneme_inventory))
    for i in range(len(phoneme_inventory)):
        total = confusion[:, i].sum()
        if total > 0:
            acc_per_phoneme[i] = confusion[i, i] / total

    overall_acc = total_correct / (total_length + 1e-6)
    return confusion, acc_per_phoneme, overall_acc

def report(confusion, acc_per_phoneme, overall_acc, title=''):
    phonemes = phoneme_inventory
    print(f"\n====== {title} ======")
    print(f"\n[Overall Phoneme Accuracy]: {overall_acc * 100:.2f}%")

    phoneme_totals = confusion.sum(axis=0)
    misclassified_counts = confusion.sum(axis=0) - np.diag(confusion)

    # 1. Top confused phoneme pairs (by count)
    pair_freq_list = []
    for i in range(len(phonemes)):
        for j in range(len(phonemes)):
            if i == j:
                continue
            count = confusion[i, j]
            total = phoneme_totals[j]
            if count > 0 and total > 0:
                error_rate = count / total
                pair_freq_list.append((count, error_rate, i, j, total))

    pair_freq_list.sort(key=lambda x: x[0], reverse=True)
    print("\n[Top 10 Confused Phoneme Pairs (by count)]")
    for idx, (count, err, i, j, total) in enumerate(pair_freq_list[:10]):
        print(f"{idx+1}. {phonemes[i]} → {phonemes[j]}: {err*100:.2f}% ({count}/{int(total)})")

    # 2. Top confused phoneme pairs (by error rate)
    pair_error_list = [(err, i, j, count, total)
                       for count, err, i, j, total in pair_freq_list]
    pair_error_list.sort(key=lambda x: x[0], reverse=True)
    print("\n[Top 10 Confused Phoneme Pairs (by error rate)]")
    for idx, (err, i, j, count, total) in enumerate(pair_error_list[:10]):
        print(f"{idx+1}. {phonemes[i]} → {phonemes[j]}: {err*100:.2f}% ({count}/{int(total)})")

    # 3. Most misclassified phonemes (by total error count)
    misclassified_list = [(i, misclassified_counts[i], phoneme_totals[i])
                          for i in range(len(phonemes)) if phoneme_totals[i] > 0]
    misclassified_list.sort(key=lambda x: x[1], reverse=True)
    print("\n[Top 10 Most Misclassified Phonemes (by count)]")
    for idx, (i, wrong, total) in enumerate(misclassified_list[:10]):
        err_rate = wrong / total
        print(f"{idx+1}. {phonemes[i]}: {err_rate*100:.2f}% ({int(wrong)}/{int(total)})")

    # 4. Lowest accuracy phonemes (normalized)
    filtered_acc = [(i, acc_per_phoneme[i], phoneme_totals[i])
                    for i in range(len(phonemes)) if phoneme_totals[i] > 0]
    filtered_acc.sort(key=lambda x: x[1])
    print("\n[Top 10 Lowest Accuracy Phonemes (normalized)]")
    for idx, (i, acc, total) in enumerate(filtered_acc[:10]):
        err = 1.0 - acc
        wrong = int(err * total)
        print(f"{idx+1}. {phonemes[i]}: {err*100:.2f}% ({wrong}/{int(total)})")
        

@hydra.main(version_base=None, config_path="../configs", config_name="emg2feat")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.ckpt_epoch is None or cfg.ckpt_epoch == 'last':
        checkpoint_path = os.path.join(cfg.exp_path, cfg.exp_name, 'last.ckpt')
    elif cfg.ckpt_epoch == 'best':
        from preprocess.emg2feat import get_best_ckpt
        checkpoint_path = get_best_ckpt(cfg.exp_name)
    else:
        checkpoint_path = glob(os.path.join(cfg.exp_path, cfg.exp_name, f'epoch={cfg.ckpt_epoch:02d}*'))[0]

    print(f"Loading checkpoint from {checkpoint_path}")
    model = EMGEncoder.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()

    feature = model.hparams.feature_config
    dataset_path = os.path.join(cfg.data_path, 'preprocessed', 'target_feature', feature.target,
                                f'{feature.sub_option}{feature[feature.sub_option]}')


    #for subset in ['voiced_parallel_data', 'silent_parallel_data']:
    for subset in ['silent_parallel_data']:
        dataset = EMGDataset(base_dir=dataset_path,
                             target=feature.target,
                             data_split=cfg.split,
                             frame_rate=feature.frame_rate,
                             normalize=feature.normalize)

        dataset.data_path = [p for p in dataset.data_path if subset in p]

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_to_max_len)

        confusion, acc_per_phoneme, overall_acc = cal_ph_acc(model, dataloader, device)
        #report(confusion, acc_per_phoneme, overall_acc, title=f"{cfg.split.upper()} - {subset}")
        save_phoneme_analysis_csv(confusion, acc_per_phoneme, csv_path=f'phoneme_analysis/{cfg.exp_name}_{cfg.ckpt_epoch}.csv')


if __name__ == "__main__":
    main()
