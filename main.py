import pytorch_lightning as pl
from glob import glob
import os
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
import pickle
import torch

from loader import EMGDataset
from modules import EMGEncoder
from data_utils import phoneme_inventory
from utils import load_partial_pretrained_model

WANDB_ID = 'dlswns8'
class AlwaysSaveLast(Callback):
    """Always overwrite a single 'last.ckpt' at each train epoch end."""
    def __init__(self, dirpath: str, filename: str = "last.ckpt", every_n_epochs: int = 1):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.every_n_epochs = int(every_n_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.every_n_epochs <= 0:
            return
        epoch1 = int(trainer.current_epoch) + 1
        if epoch1 % self.every_n_epochs != 0:
            return
        os.makedirs(self.dirpath, exist_ok=True)
        path = os.path.join(self.dirpath, self.filename)
        trainer.save_checkpoint(path)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg:DictConfig):
    log = logging.getLogger(__name__)
    log.info("Info level message")
    log.debug("Debug level message")
    log.info(OmegaConf.to_yaml(cfg))
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception as e:
        log.warning(f"Failed to set matmul precision: {e}")
    # Reproducibility (also seeds dataloader workers)
    if "seed" in cfg and cfg.seed is not None:
        pl.seed_everything(int(cfg.seed), workers=True)
    exp_dir = os.path.join(cfg.exp_path, cfg.exp_name)
    fig_dir = os.path.join(exp_dir, 'figure') if 'mspec' in cfg.feature.target else None
    preprocessed_dir = os.path.join(cfg.paths.data_path, 'preprocessed', 'target_feature', cfg.feature.target, f'{cfg.feature.sub_option}{cfg["feature"][cfg.feature.sub_option]}')
    log.info(exp_dir)
    
    if cfg.debug:
        wandb_logger = None
        feat_norm = None
        fig_dir = None
    else:
        wandb_logger = WandbLogger(project='KLEC_channel', entity=WANDB_ID, name=cfg.exp_name, save_dir=cfg.exp_path)
        if fig_dir != None:
            os.makedirs(fig_dir, exist_ok=True)
            feat_norm, _ = pickle.load(open(os.path.join(preprocessed_dir, 'normalizer.pkl'), 'rb'))
        else:
            feat_norm = None

    # Always create exp dir + save merged config (even in debug mode)
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "merged_config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    best_ckpt = ModelCheckpoint(
        dirpath=exp_dir,
        filename="{epoch:02d}-{val_phone_accuracy_best:.4f}",
        monitor="val_phone_accuracy_best",
        mode="max",
        save_top_k=3,
        save_last=False,
        every_n_epochs=5,
        auto_insert_metric_name=False,
        verbose=True,
    )
    last_ckpt = AlwaysSaveLast(exp_dir, filename="last.ckpt", every_n_epochs=1)

    if cfg.ckpt_epoch != None:
        ckpt_path = glob('%s/epoch=%02d*'%(exp_dir, cfg.ckpt_epoch))[0]
        model = EMGEncoder(cfg.emg_enc, cfg.optimizer, cfg.feature, len(phoneme_inventory), cfg.phoneme_loss_weight, cfg.batch_size, fig_dir, feat_norm)
        print(ckpt_path)
    else:
        ckpt_path = None
        model = EMGEncoder(cfg.emg_enc, cfg.optimizer, cfg.feature, len(phoneme_inventory), cfg.phoneme_loss_weight, cfg.batch_size, fig_dir, feat_norm)

    if cfg.pretrained_model != None:
        pretrained_path = glob('%s/%s/epoch=%02d*'%(cfg.exp_path, cfg.pretrained_model, cfg.pretrained_epoch))[0]
        model = load_partial_pretrained_model(model, pretrained_path, cfg.emg_enc.use_channel)

    #early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=10, verbose=False, mode='min')

    trainer_config = cfg.trainer

    trainer = pl.Trainer(logger=wandb_logger, callbacks=[best_ckpt, last_ckpt], **trainer_config)

    trainset = EMGDataset(preprocessed_dir, cfg.feature.target, 'train', cfg.feature.frame_rate, cfg.target_sec, cfg.feature.normalize)
    train_loader = DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=bool(cfg.num_workers and cfg.num_workers > 0),
        collate_fn=trainset.collate_to_max_len,
    )

    validset = EMGDataset(preprocessed_dir, cfg.feature.target, 'valid', cfg.feature.frame_rate, target_sec=None, normalize=cfg.feature.normalize)
    valid_loader = DataLoader(
        validset,
        batch_size=1,
        num_workers=min(4, int(cfg.num_workers)),
        shuffle=False,
        pin_memory=True,
        persistent_workers=bool(cfg.num_workers and cfg.num_workers > 0),
        collate_fn=validset.collate_to_max_len,
    )


    trainer.fit(model, train_loader, val_dataloaders=[valid_loader], ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()


