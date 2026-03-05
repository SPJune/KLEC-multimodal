from glob import glob
import hydra
import librosa 
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import sys
import torch
from tqdm import tqdm

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from data_utils import mel_spectrogram

def find_zero_output(model, cfg):
    if cfg.group == 'mspec':
        audio = torch.zeros(1, cfg.sr, dtype=torch.float32)
        x = mel_spectrogram(audio, cfg.nfft, cfg.dim, cfg.sr, cfg.hop, cfg.nfft, 0, cfg.sr//2, center=False)
        x = x[0].T
    else:
        print("configuration error")
        sys.exit()
    x = x.numpy()
    x = np.mean(x, 0)
    return x

@hydra.main(version_base=None, config_path="../configs", config_name="speech2feat")
def main(cfg:DictConfig):
    sr16 = 16000
    data_dir = os.path.join(cfg.data_path, 'silent_speech_dataset')
    output_dir = os.path.join(cfg.data_path, 'preprocessed', 'target_feature', cfg.feature.target, f'{cfg.feature.sub_option}{cfg["feature"][cfg.feature.sub_option]}')
    print(OmegaConf.to_yaml(cfg))
    print(output_dir)
    print(cfg.feature.group)

    if cfg.feature.group == 'contentvec':
        import fairseq
        ckpt_path = os.path.join(cfg.exp_path, "fairseq_pretrained/contentvec_legacy_100.pt")
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        model = models[0].eval().cuda()
    elif cfg.feature.group == 'mspec':
        sr_out = cfg.feature.sr
        model = None
    else:
        print(f"Undefined feature: {cfg.feature}")

    zero_output = find_zero_output(model, cfg.feature)
    os.makedirs(output_dir, exist_ok=True)
    zero_output_path = os.path.join(output_dir, 'zero_output.npy')
    np.save(zero_output_path, zero_output)

    mean_vector = np.zeros(cfg.feature.dim)
    total_count = 0
    M2 = np.zeros(cfg.feature.dim)
    for root, dirs, files in tqdm(os.walk(data_dir)):
        if len(files) > 0 and 'silent/' not in root:
            for file_name in files:
                if '.flac' in file_name:
                    org_path = os.path.join(root, file_name)
                    audio, sr = librosa.load(org_path, sr=None)
                    if cfg.feature.group == 'contentvec':
                        assert sr == sr16
                        if sr != sr16:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=sr16)
                        with torch.no_grad():
                            x = torch.tensor(audio).unsqueeze(0)
                            x = x.cuda()
                            feat_chunk, _ = model.extract_features(source=x, padding_mask=None, mask=False, output_layer=args.layer)
                        x = feat_chunk[0].cpu()
                    elif cfg.feature.group == 'mspec':
                        sr_out = cfg.feature.sr
                        if sr != sr_out:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_out)
                        x = mel_spectrogram(torch.tensor(audio, dtype=torch.float32).unsqueeze(0), cfg.feature.nfft, cfg.feature.dim, sr_out, cfg.feature.hop, cfg.feature.nfft, 0, 8000, center=False)
                        x = x[0].T
                    else:
                        print("configuration error")
                        sys.exit()
                    x = x.numpy()

                    for vectors in x:
                        for vector in vectors:
                            total_count += 1
                            delta = vector - mean_vector
                            mean_vector += delta / total_count
                            delta2 = vector - mean_vector
                            M2 += delta * delta2

                    path_save_dir = root.replace(data_dir, output_dir)
                    path_save = os.path.join(path_save_dir, file_name.replace('audio_clean.flac','feat.npy'))
                    os.makedirs(path_save_dir, exist_ok=True)
                    np.save(path_save, x)
    print(f'Total count: {total_count}')
    variance = M2 / total_count
    std_dev = np.sqrt(variance)
    normalizer_path = os.path.join(output_dir, 'normalizer.npz')
    np.savez(normalizer_path, mean=mean_vector, std=std_dev)



if __name__ == '__main__':
    main()


