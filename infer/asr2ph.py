import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os
from phonemizer.phonemize import phonemize
import editdistance

def text2phoneme(text):
    return phonemize(text, language='en-us', backend='espeak', strip=True)


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg:DictConfig):
    data_dir = os.path.join(cfg.exp_path, cfg.wav_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger_name = 'phoneme.log'
    file_handler = logging.FileHandler(os.path.join(data_dir, logger_name), mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    with open(os.path.join(data_dir, 'asr.log'), "r", encoding="utf-8") as file:  
        texts = file.readlines()

    diffs = 0
    Ls = 0
    for i in range(199):
        sess, num = texts[i*3].split()[1], texts[i*3].split()[2][:-1]
        gt = texts[i*3].split()[3:]
        gt = ' '.join(gt)
        est = texts[i*3+1].split()[3:]
        est = ' '.join(est)
        gt = text2phoneme(gt)
        est = text2phoneme(est)
        diff = editdistance.eval(gt, est)
        L = len(gt)
        diffs += diff
        Ls += L
        per0 = diff/L
        logger.info(f'trgt{i} {sess} {i}: {gt}')
        logger.info(f'pred{i} {sess} {i}: {est}')
        logger.info(f'{i} sample per: {per0*100:.2f}')
    logger.info(f'PER: {diffs/Ls*100:.2f}%')

if __name__ == '__main__':
    main()
	

