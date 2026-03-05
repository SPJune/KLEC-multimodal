import hydra
import json
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import os
import pytorch_lightning as pl
import random
import math
from torch import nn
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from typing import Optional
import sys

from loss import dtw_loss

from hifi_gan.env import AttrDict
from hifi_gan.models import Generator

V2SFLOW_ENCODER_CKPT_DEFAULT = "/data2/spjune/v2sflow/v2sflow_encoder.pt"
HIFIGAN_VOCODER_CKPT_DEFAULT = "/data2/spjune/v2sflow/hifigan_vocoder.pt"


def _linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class V2SFlowContentEncoder(nn.Module):
    def __init__(
        self,
        encoder_embed_dim: int = 1024,
        conformer_layers: int = 12,
        conformer_embed_dim: int = 512,
        conformer_ffn_embed_dim: int = 2048,
        conformer_attention_heads: int = 8,
        conformer_dropout: float = 0.1,
        conformer_attention_dropout: float = 0.1,
        conformer_layer_norm_first: bool = True,
        content_vocab_size: int = 1000,
        output_type: str = "hidden",  # ids|logits|hidden
    ):
        super().__init__()
        project_root = os.path.dirname(os.path.abspath(__file__))
        v2sflow_root = os.path.join(project_root, "V2SFlow")
        if v2sflow_root not in sys.path:
            sys.path.insert(0, v2sflow_root)
        from third_party.cosyvoice.transformer.encoder import ConformerEncoder

        self.conformer = ConformerEncoder(
            input_size=encoder_embed_dim,
            output_size=conformer_embed_dim,
            attention_heads=conformer_attention_heads,
            linear_units=conformer_ffn_embed_dim,
            num_blocks=conformer_layers,
            dropout_rate=conformer_dropout,
            positional_dropout_rate=conformer_dropout,
            attention_dropout_rate=conformer_attention_dropout,
            normalize_before=conformer_layer_norm_first,
            input_layer="linear",
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            use_cnn_module=True,
            macaron_style=True,
            cnn_module_kernel=31,
        )

        self.content_vocab_size = int(content_vocab_size)
        embed_dim = 4 + self.content_vocab_size
        self.output_type = str(output_type)
        self.conformer_embed_dim = int(conformer_embed_dim)
        self.logit_dim = int(embed_dim)
        self.proj_out = _linear(conformer_embed_dim, embed_dim) if conformer_embed_dim != embed_dim else None

    def forward(self, video_feat: torch.Tensor, video_padding_mask: torch.Tensor) -> torch.Tensor:
        if video_feat.size(1) == 0:
            B, _, C = video_feat.shape
            video_feat = torch.zeros((B, 1, C), device=video_feat.device, dtype=video_feat.dtype)
            video_padding_mask = torch.ones((B, 1), device=video_padding_mask.device, dtype=video_padding_mask.dtype)

        x = video_feat.repeat_interleave(2, dim=1)               # (B, 2T, 1024)
        padding_mask = video_padding_mask.repeat_interleave(2, dim=1)  # (B, 2T)
        lengths = (~padding_mask).sum(dim=1)
        if (lengths == 0).any():
            zero = lengths == 0
            padding_mask = padding_mask.clone()
            padding_mask[zero, 0] = False
            lengths = lengths.clone()
            lengths[zero] = 1

        h, _ = self.conformer(x, lengths)  # (B, 2T, conformer_embed_dim)

        if self.output_type == "hidden":
            return h

        logits = self.proj_out(h) if self.proj_out is not None else h  # (B, 2T, 4+vocab) when proj_out exists
        if self.output_type == "logits":
            return logits

        if self.output_type == "ids":
            logits = logits.clone()
            logits[..., :4] = -math.inf
            ids = logits.argmax(dim=-1) - 4  # (B, 2T), range [0, vocab-1]
            return ids

        raise ValueError(f"Unknown output_type: {self.output_type}")


class V2SFlowVideoEncoder(nn.Module):

    def __init__(self, content_vocab_size: int = 1000, output_type: str = "hidden"):
        super().__init__()
        self.content_encoder = V2SFlowContentEncoder(content_vocab_size=content_vocab_size, output_type=output_type)

    def forward(self, video_feat: torch.Tensor, video_padding_mask: torch.Tensor) -> torch.Tensor:
        return self.content_encoder(video_feat=video_feat, video_padding_mask=video_padding_mask)


def _load_state_dict_flexible(model: nn.Module, ckpt_path: str, strict: bool = False):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint type at {ckpt_path}: {type(state)}")
    return model.load_state_dict(state, strict=strict)


def topk_masking(mask, k):
    topk_vals, topk_indices = torch.topk(mask, k)
    masked = torch.zeros_like(mask)
    masked[topk_indices] = topk_vals
    return masked

class ChannelDropout(nn.Module):
    def __init__(self, drop_prob=0.3):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):  # x: (B, T, C)
        if not self.training or self.drop_prob == 0:
            return x
        mask = (torch.rand(x.shape[-1]) > self.drop_prob).float().to(x.device)
        return x * mask

class Vocoder(object):
    def __init__(self, checkpoint_file=HIFIGAN_VOCODER_CKPT_DEFAULT, \
                 device='cuda', half=False):
        config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
        with open(config_file) as f:
            hparams = AttrDict(json.load(f))
        self.generator = Generator(hparams).to(device)
        ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "generator" in ckpt:
            gen_state = ckpt["generator"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            gen_state = ckpt["state_dict"]
            if any(k.startswith("generator.") for k in gen_state.keys()):
                gen_state = {k[len("generator."):]: v for k, v in gen_state.items() if k.startswith("generator.")}
        elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
            gen_state = ckpt
        else:
            raise TypeError(f"Unsupported HiFi-GAN checkpoint format: {type(ckpt)}")
        self.generator.load_state_dict(gen_state, strict=False)
        self.generator.eval()
        if half:
            self.generator.half()
        self.generator.remove_weight_norm()

    def __call__(self, mel_spectrogram):
        with torch.no_grad():
            mel_spectrogram = mel_spectrogram.T[np.newaxis,:,:]
            audio = self.generator(mel_spectrogram)
        return audio.squeeze()
        
class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)

class EMGEncoder(pl.LightningModule):
    def __init__(self, emg_enc_config: DictConfig, optimizer_config: DictConfig, feature_config: DictConfig, num_aux_outs=None, phoneme_loss_weight=0.5, batch_size=128, fig_dir=None, feat_norm=None):
        super(EMGEncoder, self).__init__()
        self.save_hyperparameters("emg_enc_config", "optimizer_config", "feature_config", "num_aux_outs", "phoneme_loss_weight", "batch_size")
        model_size = emg_enc_config.model_size
        dropout = emg_enc_config.dropout
        num_layers = int(emg_enc_config.num_layers)
        self.optimizer_config = optimizer_config
        num_outs = feature_config.dim
        self.learning_rate_warmup = optimizer_config.lr_warmup
        self.batch_size = batch_size
        emg_num_ch = len(emg_enc_config.use_channel)
        self.emg_ch = emg_enc_config.use_channel
        self.channel_dropout = ChannelDropout(emg_enc_config.get('channel_dropout', 0))

        self.conv_blocks = nn.Sequential(
            ResBlock(emg_num_ch, model_size, 2),
            ResBlock(model_size, model_size, 2),
            ResBlock(model_size, model_size, 1),
        )
        self.w_raw_in = nn.Linear(model_size, model_size)

        #encoder_layer = nn.TransformerEncoderLayer(d_model=model_size, nhead=8, dim_feedforward=3072, dropout=dropout)
        encoder_layer = TransformerEncoderLayer(d_model=model_size, nhead=8, relative_positional=True, relative_positional_distance=100, dim_feedforward=3072, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.transformer_num_layers = num_layers
        self.w_out = nn.Linear(model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(model_size, num_aux_outs)
            # Video heads (same structure as EMG heads)
            self.wv_out = nn.Linear(model_size, num_outs)
            self.wv_aux = nn.Linear(model_size, num_aux_outs)

        # ------------------------------------------------------------
        # Video branch (V2SFlow encoder content head)
        # ------------------------------------------------------------
        self.use_video = bool(emg_enc_config.get("use_video", True))
        # debug flags (print once at step0)
        self.debug_grad = bool(emg_enc_config.get("debug_grad", False))
        self._debug_printed = False
        self._debug_video_stats = None
        # emg_only|video_only|both
        self.modality = str(emg_enc_config.get("modality", "both"))
        # concat|ca|ab
        self.fusion_method = str(emg_enc_config.get("fusion_method", "concat"))
        # where to fuse inside transformer stack (0..num_layers)
        self.fusion_after_layer = int(emg_enc_config.get("fusion_after_layer", 0))
        if self.fusion_after_layer < 0 or self.fusion_after_layer > self.transformer_num_layers:
            raise ValueError(
                f"fusion_after_layer must be in [0, {self.transformer_num_layers}], got {self.fusion_after_layer}"
            )

        self.video_encoder_mode = str(emg_enc_config.get("video_encoder_mode", "freeze"))  # freeze|finetune|scratch
        # ids: original argmax ids (non-differentiable)
        # logits: use logits -> soft embedding (differentiable, enables finetune)
        # hidden: use conformer hidden -> linear proj (differentiable, enables finetune)
        self.video_encoder_output = str(emg_enc_config.get("video_encoder_output", "hidden"))  # ids|logits|hidden
        self.video_encoder_ckpt = str(emg_enc_config.get("video_encoder_ckpt", V2SFLOW_ENCODER_CKPT_DEFAULT))
        self.video_content_vocab_size = int(emg_enc_config.get("video_content_vocab_size", 1000))

        if self.use_video:
            self.video_encoder = V2SFlowVideoEncoder(
                content_vocab_size=self.video_content_vocab_size,
                output_type=self.video_encoder_output,
            )
            if self.video_encoder_mode != "scratch":
                incompatible = _load_state_dict_flexible(self.video_encoder, self.video_encoder_ckpt, strict=False)
                if self.debug_grad:
                    missing = getattr(incompatible, "missing_keys", None)
                    unexpected = getattr(incompatible, "unexpected_keys", None)
                    print(
                        f"[debug] video_encoder ckpt loaded: missing={len(missing) if missing is not None else 'NA'} "
                        f"unexpected={len(unexpected) if unexpected is not None else 'NA'} "
                        f"mode={self.video_encoder_mode} output={self.video_encoder_output}"
                    )
            if self.video_encoder_mode == "freeze":
                for p in self.video_encoder.parameters():
                    p.requires_grad = False
            elif self.video_encoder_mode in ("finetune", "scratch"):
                pass
            else:
                raise ValueError(f"Unknown video_encoder_mode: {self.video_encoder_mode}")

            # content id sequence -> embedding
            self.video_content_embed = nn.Embedding(self.video_content_vocab_size, model_size)
            # hidden -> model embedding
            self.video_hidden_proj = nn.Linear(getattr(self.video_encoder.content_encoder, "conformer_embed_dim", 512), model_size)
            self.video_to_emg = nn.Linear(model_size, model_size)

        # ------------------------------------------------------------
        # Fusion blocks
        # ------------------------------------------------------------

        if self.fusion_method == "concat":
            self.fusion_proj = nn.Linear(model_size * 2, model_size)
        elif self.fusion_method == "high_concat":
            # Late fusion: concat after full EMG transformer (right before heads)
            self.fusion_high_proj = nn.Linear(model_size * 2, model_size)
        elif self.fusion_method == "ca":
            num_heads = int(emg_enc_config.get("fusion_num_heads", 8))
            self.fusion_ca = nn.MultiheadAttention(model_size, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.fusion_ca_ln = nn.LayerNorm(model_size)
        elif self.fusion_method == "ab":
            self.ab_num_tokens = int(emg_enc_config.get("ab_num_tokens", 8))
            self.ab_tokens = nn.Parameter(torch.randn(1, self.ab_num_tokens, model_size) * 0.02)
            # Paper-style Attention Bottleneck uses modality-specific Transformer params (θ_i).
            # We implement fusion layers (from fusion_after_layer .. num_layers-1) as:
            #   [z_i || z_fsn] -> Transformer_i -> [z_i' || zhat_fsn_i]
            #   z_fsn <- Avg_i(zhat_fsn_i)
            num_heads = int(emg_enc_config.get("fusion_num_heads", 8))
            ab_layers = max(0, self.transformer_num_layers - self.fusion_after_layer)
            self.ab_layers_emg = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=model_size,
                        nhead=num_heads,
                        dim_feedforward=3072,
                        dropout=dropout,
                        batch_first=True,
                        activation="relu",
                    )
                    for _ in range(ab_layers)
                ]
            )
            self.ab_layers_vid = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=model_size,
                        nhead=num_heads,
                        dim_feedforward=3072,
                        dropout=dropout,
                        batch_first=True,
                        activation="relu",
                    )
                    for _ in range(ab_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

        self.loss_fn = dtw_loss
        self.phoneme_loss_weight = phoneme_loss_weight
        self.fig_dir = fig_dir
        self.feat_norm = feat_norm
        # Debug: print tensor shapes once when SHAPE_DEBUG=1
        self.shape_debug = os.environ.get("SHAPE_DEBUG", "0") == "1"
        self._shape_debug_printed = False

    def _run_transformer_layers(self, x_btd: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Run a slice of the encoder layers [start:end] on x (B,T,D)."""
        if start >= end:
            return x_btd
        x = x_btd.transpose(0, 1)  # (T,B,D)
        # nn.TransformerEncoder stores layers in .layers
        for i in range(int(start), int(end)):
            x = self.transformer.layers[i](x)
        x = x.transpose(0, 1)  # (B,T,D)
        return x

    def forward(self, x_raw, video_feat: Optional[torch.Tensor] = None, video_padding_mask: Optional[torch.Tensor] = None):
        # ------------------------------------------------------------
        # 1) Build modality sequences (B, T, D)
        # ------------------------------------------------------------
        if self.shape_debug and (not self._shape_debug_printed):
            try:
                print(
                    "[shape][forward:in] "
                    f"x_raw{tuple(x_raw.shape)} "
                    f"video_feat{(tuple(video_feat.shape) if video_feat is not None else None)} "
                    f"video_padding_mask{(tuple(video_padding_mask.shape) if video_padding_mask is not None else None)} "
                    f"modality={self.modality} fusion={self.fusion_method} fusion_after_layer={self.fusion_after_layer} use_video={self.use_video} "
                    f"video_out={self.video_encoder_output}"
                )
            except Exception as e:
                print("[shape][forward:in] failed:", repr(e))

        emg_seq = None
        if self.modality in ("emg_only", "both"):
            # x_raw shape is (batch, time, electrode)
            emg = x_raw[:, :, self.emg_ch]
            emg = self.channel_dropout(emg)
            emg = emg.transpose(1, 2)  # channel before time for conv
            emg = self.conv_blocks(emg)
            emg = emg.transpose(1, 2)
            emg_seq = self.w_raw_in(emg)  # (B, T, D)
            if self.shape_debug and (not self._shape_debug_printed):
                try:
                    print(f"[shape][forward:emg] conv_out{tuple(emg.shape)} emg_seq{tuple(emg_seq.shape)}")
                except Exception as e:
                    print("[shape][forward:emg] failed:", repr(e))

        video_seq = None
        video_pad_100 = None
        if (self.use_video and (video_feat is not None)) and (self.modality in ("video_only", "both")):
            if video_padding_mask is None:
                video_padding_mask = torch.zeros(
                    video_feat.shape[0],
                    video_feat.shape[1],
                    dtype=torch.bool,
                    device=video_feat.device,
                )

            # V2SFlow content encoder output @ 50Hz (B, 2T_vid, *)
            content_out = self.video_encoder(video_feat=video_feat, video_padding_mask=video_padding_mask)
            if self.shape_debug and (not self._shape_debug_printed):
                try:
                    print(f"[shape][forward:video] content_out{tuple(content_out.shape)}")
                except Exception as e:
                    print("[shape][forward:video] failed:", repr(e))
            if self.video_encoder_output == "ids":
                # (B, 2T_vid) int64
                content_emb = self.video_content_embed(
                    content_out.clamp(min=0, max=self.video_content_vocab_size - 1)
                )  # (B, 2T_vid, D)
            elif self.video_encoder_output == "logits":
                # (B, 2T_vid, 4+vocab) float
                logits = content_out
                # mask specials and soft-embed over vocab
                logits = logits.clone()
                logits[..., :4] = -torch.inf
                probs = torch.softmax(logits, dim=-1)[..., 4:]  # (B, 2T_vid, vocab)
                content_emb = probs @ self.video_content_embed.weight  # (B, 2T_vid, D)
            elif self.video_encoder_output == "hidden":
                # (B, 2T_vid, conformer_dim) float
                content_emb = self.video_hidden_proj(content_out)  # (B, 2T_vid, D)
            else:
                raise ValueError(f"Unknown video_encoder_output: {self.video_encoder_output}")


            content_emb = content_emb.repeat_interleave(2, dim=1)
            video_pad_100 = video_padding_mask.repeat_interleave(4, dim=1)  # True=pad
            if self.shape_debug and (not self._shape_debug_printed):
                try:
                    print(f"[shape][forward:video] content_emb_100Hz{tuple(content_emb.shape)} video_pad_100{tuple(video_pad_100.shape)}")
                except Exception as e:
                    print("[shape][forward:video] failed:", repr(e))

            # Time alignment (to EMG length if present, else keep its own length)
            if emg_seq is not None:
                T = emg_seq.shape[1]
            else:
                T = content_emb.shape[1]

            if content_emb.shape[1] >= T:
                content_emb = content_emb[:, :T, :]
                video_pad_100 = video_pad_100[:, :T]
            else:
                pad_len = T - content_emb.shape[1]
                content_emb = torch.cat(
                    [
                        content_emb,
                        torch.zeros(
                            content_emb.shape[0],
                            pad_len,
                            content_emb.shape[2],
                            device=content_emb.device,
                            dtype=content_emb.dtype,
                        ),
                    ],
                    dim=1,
                )
                video_pad_100 = torch.cat(
                    [
                        video_pad_100,
                        torch.ones(
                            video_pad_100.shape[0],
                            pad_len,
                            device=video_pad_100.device,
                            dtype=video_pad_100.dtype,
                        ),
                    ],
                    dim=1,
                )

            content_emb = content_emb * (~video_pad_100).unsqueeze(-1).to(dtype=content_emb.dtype)
            video_seq = self.video_to_emg(content_emb)  # (B, T, D)
            if self.shape_debug and (not self._shape_debug_printed):
                try:
                    print(f"[shape][forward:video] video_seq{tuple(video_seq.shape)} (aligned_T={int(video_seq.shape[1])})")
                except Exception as e:
                    print("[shape][forward:video] failed:", repr(e))

        # If model expects video but none is provided (e.g., some inference scripts),
        # fall back to "missing video" (all padded) for modality=both.
        if self.modality == "both" and self.use_video and (video_feat is None):
            if emg_seq is None:
                raise ValueError("modality=both인데 emg 입력이 비었습니다.")
            video_seq = torch.zeros_like(emg_seq)
            video_pad_100 = torch.ones(
                (video_seq.shape[0], video_seq.shape[1]),
                dtype=torch.bool,
                device=video_seq.device,
            )

        if self.modality == "video_only" and video_seq is None:
            raise ValueError("modality=video_only지만 video_feat가 제공되지 않았습니다.")
        if self.modality == "emg_only" and emg_seq is None:
            raise ValueError("modality=emg_only지만 emg 입력이 비었습니다.")

        # ------------------------------------------------------------
        # 2) Fuse (B, T, D)
        # ------------------------------------------------------------
        vid_out = None
        if self.modality == "emg_only":
            x = self._run_transformer_layers(emg_seq, 0, self.transformer_num_layers)
        elif self.modality == "video_only":
            x = self._run_transformer_layers(video_seq, 0, self.transformer_num_layers)
        else:
            # both
            if self.fusion_method == "high_concat":
                # EMG baseline (full): conv + transformer, then concat with video right before heads.
                emg_high = self._run_transformer_layers(emg_seq, 0, self.transformer_num_layers)
                x = self.fusion_high_proj(torch.cat([emg_high, video_seq], dim=-1))
            else:
                # EMG baseline: conv + (pre-fusion transformer layers)
                emg_pre = self._run_transformer_layers(emg_seq, 0, self.fusion_after_layer)
                if self.fusion_method == "concat":
                    x = self.fusion_proj(torch.cat([emg_pre, video_seq], dim=-1))
                elif self.fusion_method == "ca":
                    # Q=emg, K/V=video
                    attn_out, _ = self.fusion_ca(
                        query=emg_pre,
                        key=video_seq,
                        value=video_seq,
                        key_padding_mask=video_pad_100,
                        need_weights=False,
                    )
                    x = self.fusion_ca_ln(emg_pre + attn_out)
                elif self.fusion_method == "ab":
                    # ------------------------------------------------------------
                    # Attention Bottleneck (paper-style; Eq. 7-9) with modality-specific params (θ_i)
                    # ------------------------------------------------------------
                    B, T, D = emg_pre.shape
                    M = int(self.ab_tokens.shape[1])
                    fsn = self.ab_tokens.expand(B, -1, -1)  # z_fsn^0: (B, M, D)

                    emg_cur = emg_pre
                    vid_cur = video_seq

                    # video padding mask (True=pad). For bottleneck tokens: always False.
                    if video_pad_100 is None:
                        vid_mask = None
                        has_video = torch.ones((B, 1, 1), device=emg_pre.device, dtype=emg_pre.dtype)
                    else:
                        vid_mask = video_pad_100
                        has_video = (~video_pad_100).any(dim=1).to(dtype=emg_pre.dtype).view(B, 1, 1)  # 1 if any valid frame
                        # keep padded frames as zeros
                        vid_cur = vid_cur * (~video_pad_100).unsqueeze(-1).to(dtype=vid_cur.dtype)

                    for li, (layer_emg, layer_vid) in enumerate(zip(self.ab_layers_emg, self.ab_layers_vid)):
                        # ---- modality: EMG ----
                        emg_cat = torch.cat([emg_cur, fsn], dim=1)  # (B, T+M, D)
                        emg_cat = layer_emg(emg_cat)  # (B, T+M, D)
                        emg_cur = emg_cat[:, :T, :]
                        fsn_hat_emg = emg_cat[:, T:, :]  # (B, M, D)

                        # ---- modality: Video ----
                        vid_cat = torch.cat([vid_cur, fsn], dim=1)  # (B, T+M, D)
                        if vid_mask is None:
                            vid_cat = layer_vid(vid_cat)
                        else:
                            fsn_mask = torch.zeros((B, M), dtype=torch.bool, device=vid_mask.device)
                            cat_mask = torch.cat([vid_mask, fsn_mask], dim=1)  # (B, T+M)
                            vid_cat = layer_vid(vid_cat, src_key_padding_mask=cat_mask)
                        vid_cur = vid_cat[:, :T, :]
                        fsn_hat_vid = vid_cat[:, T:, :]  # (B, M, D)

                        # Eq. (9): average temporary bottlenecks (ignore video if missing)
                        denom = 1.0 + has_video
                        fsn = (fsn_hat_emg + has_video * fsn_hat_vid) / denom

                        # keep padded video frames as zeros
                        if vid_mask is not None:
                            vid_cur = vid_cur * (~vid_mask).unsqueeze(-1).to(dtype=vid_cur.dtype)

                    # Downstream uses the EMG stream (video influences it only via z_fsn).
                    x = emg_cur
                    vid_out = vid_cur
                else:
                    raise ValueError(f"Unknown fusion_method: {self.fusion_method}")
        if self.shape_debug and (not self._shape_debug_printed):
            try:
                print(f"[shape][forward:fuse] fused_x{tuple(x.shape)}")
            except Exception as e:
                print("[shape][forward:fuse] failed:", repr(e))

        # ------------------------------------------------------------
        # 3) Post-fusion transformer + heads
        # ------------------------------------------------------------
        if self.modality == "both" and self.fusion_method in ("concat", "ca"):
            x = self._run_transformer_layers(x, self.fusion_after_layer, self.transformer_num_layers)

        feat = self.w_out(x)
        ph = self.w_aux(x)
        feat_vid = None
        ph_vid = None
        if (vid_out is not None) and self.has_aux_out:
            feat_vid = self.wv_out(vid_out)
            ph_vid = self.wv_aux(vid_out)
        if self.shape_debug and (not self._shape_debug_printed):
            try:
                print(f"[shape][forward:out] feat{tuple(feat.shape)} ph{tuple(ph.shape)} feat_vid{(tuple(feat_vid.shape) if feat_vid is not None else None)} ph_vid{(tuple(ph_vid.shape) if ph_vid is not None else None)}")
            except Exception as e:
                print("[shape][forward:out] failed:", repr(e))
            self._shape_debug_printed = True
        return feat, ph, feat_vid, ph_vid

    def configure_optimizers(self):
        optimizer_class = hydra.utils.get_class(self.optimizer_config.target)
        trainable_params = (
                            list(self.conv_blocks.parameters())
                            + list(self.w_raw_in.parameters())
                            + list(self.transformer.parameters())
                            + list(self.w_out.parameters())
                            + list(self.w_aux.parameters())
                            )
        optimizer = optimizer_class(self.parameters(), **self.optimizer_config.params)
        def lr_lambda(current_step: int):
            if current_step < self.learning_rate_warmup:
                return current_step/ float(max(1, self.learning_rate_warmup))
            return 1

        scheduler_warmup = LambdaLR(optimizer, lr_lambda)
        scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        return [optimizer], [{'scheduler': scheduler_warmup, 'interval': 'step', 'frequency': 1},
                    {'scheduler': scheduler_plateau, 'monitor': 'val_loss', 'strict': False}
                    ]

    def _compute_video_aux_loss(
        self,
        batch,
        y_hat_vid,
        y_ph_hat_vid,
        y,
        y_ph,
        silents,
        target_lengths,
        est_lengths,
        *,
        phoneme_eval: bool = False,
        return_phone_acc: bool = False,
    ):
        zero = torch.zeros((), device=y.device, dtype=y.dtype)
        if y_hat_vid is None or y_ph_hat_vid is None:
            return zero, None

        vlen = batch.get("video_lengths", None)
        if vlen is None:
            return zero, None

        has_video = vlen.to(device=y_hat_vid.device) > 0
        if not bool(has_video.any().item()):
            return zero, None

        idx = torch.nonzero(has_video, as_tuple=False).squeeze(-1)
        est_vid = batch.get("video_est_lengths", est_lengths).to(device=y_hat_vid.device)
        loss_vid, _, _, phone_acc_vid = self.loss_fn(
            y_hat_vid.index_select(0, idx),
            y_ph_hat_vid.index_select(0, idx),
            y.index_select(0, idx),
            y_ph.index_select(0, idx),
            silents.index_select(0, idx),
            self.phoneme_loss_weight,
            target_lengths.index_select(0, idx),
            est_vid.index_select(0, idx),
            phoneme_eval=phoneme_eval,
        )
        if not return_phone_acc:
            phone_acc_vid = None
        return loss_vid, phone_acc_vid

    def training_step(self, batch, batch_idx):
        x = batch['emg']
        y = batch['speech_features']
        y_ph = batch['phonemes']
        silents = batch['silents']
        target_lengths = batch['target_lengths']
        est_lengths = batch['est_lengths']
        if getattr(self, "modality", "both") == "video_only":
            est_lengths = batch.get('video_est_lengths', est_lengths)
        video_feat = batch.get('video_features', None)
        video_padding_mask = batch.get('video_padding_mask', None)
        if self.shape_debug and batch_idx == 0:
            try:
                print(
                    "[shape][train_step:batch] "
                    f"emg{tuple(x.shape)} speech_features{tuple(y.shape)} phonemes{tuple(y_ph.shape)} "
                    f"silents{tuple(silents.shape)} target_lengths{tuple(target_lengths.shape)} est_lengths{tuple(est_lengths.shape)} "
                    f"video_features{(tuple(video_feat.shape) if video_feat is not None else None)} "
                    f"video_padding_mask{(tuple(video_padding_mask.shape) if video_padding_mask is not None else None)}"
                )
            except Exception as e:
                print("[shape][train_step:batch] failed:", repr(e))
        if self.debug_grad and (not self._debug_printed) and batch_idx == 0:
            with torch.no_grad():
                if video_feat is None:
                    self._debug_video_stats = {"video_feat": None}
                else:
                    # keep it cheap + robust for empty tensors
                    B = int(video_feat.shape[0])
                    T = int(video_feat.shape[1])
                    mean_abs = float(video_feat.abs().mean().detach().cpu()) if video_feat.numel() > 0 else 0.0
                    pad_mean = None
                    if video_padding_mask is not None and video_padding_mask.numel() > 0:
                        pad_mean = float(video_padding_mask.float().mean().detach().cpu())
                    self._debug_video_stats = {
                        "B": B,
                        "T_vid": T,
                        "mean_abs": mean_abs,
                        "pad_mean": pad_mean,
                        "has_any": bool(video_feat.numel() > 0),
                    }
        x = x.clone()
        r = random.randrange(8)
        if r > 0:
            temp = x[:,r:,:] # shift left r
            x[:,:-r,:] = temp.clone()
            x[:,-r:,:] = 0
        y_hat, y_ph_hat, y_hat_vid, y_ph_hat_vid = self(x, video_feat=video_feat, video_padding_mask=video_padding_mask)
        if self.shape_debug and batch_idx == 0:
            try:
                print(
                    "[shape][train_step:pred] "
                    f"y_hat{tuple(y_hat.shape)} y_ph_hat{tuple(y_ph_hat.shape)} "
                    f"y{tuple(y.shape)} y_ph{tuple(y_ph.shape)}"
                )
            except Exception as e:
                print("[shape][train_step:pred] failed:", repr(e))
        loss_emg, loss_dist, loss_ph, _ = self.loss_fn(y_hat, y_ph_hat, y, y_ph, silents, self.phoneme_loss_weight, target_lengths, est_lengths)
        loss_vid, _ = self._compute_video_aux_loss(
            batch,
            y_hat_vid,
            y_ph_hat_vid,
            y,
            y_ph,
            silents,
            target_lengths,
            est_lengths,
            phoneme_eval=False,
            return_phone_acc=False,
        )

        loss = loss_emg + loss_vid
        self.log('train_loss', loss, batch_size=self.batch_size, prog_bar=True)
        self.log('train_loss_emg', loss_emg, batch_size=self.batch_size, prog_bar=False)
        self.log('train_loss_vid', loss_vid, batch_size=self.batch_size, prog_bar=False)
        self.log('train_loss_dist', loss_dist, batch_size=self.batch_size, prog_bar=False)
        self.log('train_loss_ph', loss_ph, batch_size=self.batch_size, prog_bar=False)
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=self.batch_size)
        if batch_idx == 0 and self.fig_dir != None:
            self.save_mel_spectrogram(y, y_hat, step_type='train')
            self.save_mel_spectrogram(y, y_hat, step_type='train', normalize=False)
        return loss

    def on_after_backward(self):
        # Print grad flow once (first backward) to debug scratch/freeze/finetune differences.
        if (not getattr(self, "debug_grad", False)) or getattr(self, "_debug_printed", False):
            return
        try:
            step = int(getattr(self.trainer, "global_step", -1))
        except Exception:
            step = -1
        if step not in (-1, 0):
            return

        def _summarize_params(params):
            params = list(params)
            req = sum(int(p.requires_grad) for p in params)
            gnn = sum(int(p.grad is not None) for p in params)
            gmean = None
            norms = []
            for p in params:
                if p.grad is not None:
                    norms.append(float(p.grad.detach().data.norm(2).cpu()))
            if norms:
                gmean = float(sum(norms) / len(norms))
            return {"n": len(params), "req": req, "grad_not_none": gnn, "grad_norm_mean": gmean}

        print("[debug] on_after_backward step", step)
        print("[debug] modality", getattr(self, "modality", None), "fusion", getattr(self, "fusion_method", None))
        print("[debug] video mode/output", getattr(self, "video_encoder_mode", None), getattr(self, "video_encoder_output", None))
        if getattr(self, "_debug_video_stats", None) is not None:
            print("[debug] video batch stats", self._debug_video_stats)

        if hasattr(self, "video_encoder"):
            print("[debug] video_encoder", _summarize_params(self.video_encoder.parameters()))
        if hasattr(self, "video_hidden_proj"):
            print("[debug] video_hidden_proj", _summarize_params(self.video_hidden_proj.parameters()))
        if hasattr(self, "video_content_embed"):
            print("[debug] video_content_embed", _summarize_params(self.video_content_embed.parameters()))

        print("[debug] conv_blocks", _summarize_params(self.conv_blocks.parameters()))
        print("[debug] transformer", _summarize_params(self.transformer.parameters()))
        print("[debug] w_out", _summarize_params(self.w_out.parameters()))
        self._debug_printed = True

    def validation_step(self, batch, batch_idx):
        x = batch['emg']
        y = batch['speech_features']
        y_ph = batch['phonemes']
        silents = batch['silents']
        target_lengths = batch['target_lengths']
        est_lengths = batch['est_lengths']
        if getattr(self, "modality", "both") == "video_only":
            est_lengths = batch.get('video_est_lengths', est_lengths)
        video_feat = batch.get('video_features', None)
        video_padding_mask = batch.get('video_padding_mask', None)
        y_hat, y_ph_hat, y_hat_vid, y_ph_hat_vid = self(x, video_feat=video_feat, video_padding_mask=video_padding_mask)
        loss_emg, loss_dist, loss_ph, phone_acc = self.loss_fn(
            y_hat, y_ph_hat, y, y_ph, silents, self.phoneme_loss_weight, target_lengths, est_lengths, phoneme_eval=True
        )

        loss_vid, phone_acc_vid = self._compute_video_aux_loss(
            batch,
            y_hat_vid,
            y_ph_hat_vid,
            y,
            y_ph,
            silents,
            target_lengths,
            est_lengths,
            phoneme_eval=True,
            return_phone_acc=True,
        )

        # Checkpoint metric: use the higher one if vid metric exists.
        if phone_acc_vid is None:
            phone_acc_best = float(phone_acc)
        else:
            phone_acc_best = float(max(float(phone_acc), float(phone_acc_vid)))

        loss = loss_emg + loss_vid
        self.log('val_loss', loss, batch_size=1, prog_bar=True)
        self.log('val_loss_emg', loss_emg, batch_size=1, prog_bar=False)
        self.log('val_loss_vid', loss_vid, batch_size=1, prog_bar=False)
        self.log('val_loss_dist', loss_dist, batch_size=1)
        self.log('val_loss_ph', loss_ph, batch_size=1)
        self.log('val_phone_accuracy', phone_acc, batch_size=1, prog_bar=False)
        if phone_acc_vid is not None:
            self.log('val_phone_accuracy_vid', phone_acc_vid, batch_size=1, prog_bar=False)
        self.log('val_phone_accuracy_best', phone_acc_best, batch_size=1, prog_bar=True)
        if batch_idx == 0 and self.fig_dir != None:
            self.save_mel_spectrogram(y, y_hat, step_type='valid')
            self.save_mel_spectrogram(y, y_hat, step_type='valid', normalize=False)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['emg']
        y = batch['speech_features']
        y_ph = batch['phonemes']
        silents = batch['silents']
        target_lengths = batch['target_lengths']
        est_lengths = batch['est_lengths']
        if getattr(self, "modality", "both") == "video_only":
            est_lengths = batch.get('video_est_lengths', est_lengths)
        video_feat = batch.get('video_features', None)
        video_padding_mask = batch.get('video_padding_mask', None)
        y_hat, y_ph_hat, y_hat_vid, y_ph_hat_vid = self(x, video_feat=video_feat, video_padding_mask=video_padding_mask)
        loss_emg, _, _, _ = self.loss_fn(
            y_hat, y_ph_hat, y, y_ph, silents, self.phoneme_loss_weight, target_lengths, est_lengths, phoneme_eval=True
        )
        loss_vid, _ = self._compute_video_aux_loss(
            batch,
            y_hat_vid,
            y_ph_hat_vid,
            y,
            y_ph,
            silents,
            target_lengths,
            est_lengths,
            phoneme_eval=True,
            return_phone_acc=False,
        )

        loss = loss_emg + loss_vid
        self.log('test_loss', loss, batch_size=1)
        self.log('test_loss_emg', loss_emg, batch_size=1)
        self.log('test_loss_vid', loss_vid, batch_size=1)
        return loss

    def save_mel_spectrogram(self, y, y_hat, step_type, normalize=True):
        y_spec = y[0].detach().cpu().numpy()
        y_hat_spec = y_hat[0].detach().cpu().numpy()
        if normalize:
            y_spec = self.feat_norm.inverse(y_spec).T
            y_hat_spec = self.feat_norm.inverse(y_hat_spec).T
        else:
            y_spec = y_spec.T
            y_hat_spec = y_hat_spec.T
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].imshow(y_spec, origin='lower', aspect='auto')
        axs[0].set_title('Ground Truth Mel-Spectrogram')
        axs[1].imshow(y_hat_spec, origin='lower', aspect='auto')
        axs[1].set_title('Predicted Mel-Spectrogram')
        
        save_path = os.path.join(self.fig_dir, f'{step_type}_mel_spectrogram_{normalize}.png')
        plt.savefig(save_path)
        plt.close(fig)

class TransformerEncoderLayer(nn.Module):
    # Adapted from pytorch source
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, relative_positional=True, relative_positional_distance=100):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, relative_positional=relative_positional, relative_positional_distance=relative_positional_distance)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model=256, n_head=4, dropout=0.1, relative_positional=True, relative_positional_distance=100):
    super().__init__()
    self.d_model = d_model
    self.n_head = n_head
    d_qkv = d_model // n_head
    assert d_qkv * n_head == d_model, 'd_model must be divisible by n_head'
    self.d_qkv = d_qkv

    self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
    nn.init.xavier_normal_(self.w_q)
    nn.init.xavier_normal_(self.w_k)
    nn.init.xavier_normal_(self.w_v)
    nn.init.xavier_normal_(self.w_o)

    self.dropout = nn.Dropout(dropout)
    self.batch_first = False

    if relative_positional:
        self.relative_positional = LearnedRelativePositionalEmbedding(relative_positional_distance, n_head, d_qkv, True)
    else:
        self.relative_positional = None

  def forward(self, x):
    """Runs the multi-head self-attention layer.

    Args:
      x: the input to the layer, a tensor of shape [length, batch_size, d_model]
    Returns:
      A single tensor containing the output from this layer
    """

    q = torch.einsum('tbf,hfa->bhta', x, self.w_q)
    k = torch.einsum('tbf,hfa->bhta', x, self.w_k)
    v = torch.einsum('tbf,hfa->bhta', x, self.w_v)
    logits = torch.einsum('bhqa,bhka->bhqk', q, k) / (self.d_qkv ** 0.5)

    if self.relative_positional is not None:
        q_pos = q.permute(2,0,1,3) #bhqd->qbhd
        l,b,h,d = q_pos.size()
        position_logits, _ = self.relative_positional(q_pos.reshape(l,b*h,d))
        # (bh)qk
        logits = logits + position_logits.view(b,h,l,l)

    probs = F.softmax(logits, dim=-1)
    probs = self.dropout(probs)
    o = torch.einsum('bhqk,bhka->bhqa', probs, v)
    out = torch.einsum('bhta,haf->tbf', o, self.w_o)
    return out

class LearnedRelativePositionalEmbedding(nn.Module):
    # from https://github.com/pytorch/fairseq/pull/2225/commits/a7fb63f2b84d5b20c8855e9c3372a95e5d0ea073
    """
    This module learns relative positional embeddings up to a fixed
    maximum size. These are masked for decoder and unmasked for encoder
    self attention.
    By default the embeddings are added to keys, but could be added to
    values as well.
    Args:
        max_relative_pos (int): the maximum relative positions to compute embeddings for
        num_heads (int): number of attention heads
        embedding_dim (int): depth of embeddings
        unmasked (bool): if the attention is unmasked (for transformer encoder)
        heads_share_embeddings (bool): if heads share the same relative positional embeddings
        add_to_values (bool): compute embeddings to be added to values as well
    """

    def __init__(
            self,
            max_relative_pos: int,
            num_heads: int,
            embedding_dim: int,
            unmasked: bool = False,
            heads_share_embeddings: bool = False,
            add_to_values: bool = False):
        super().__init__()
        self.max_relative_pos = max_relative_pos
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.unmasked = unmasked
        self.heads_share_embeddings = heads_share_embeddings
        self.add_to_values = add_to_values
        num_embeddings = (
            2 * max_relative_pos - 1
            if unmasked
            else max_relative_pos
        )
        embedding_size = (
            [num_embeddings, embedding_dim, 1]
            if heads_share_embeddings
            else [num_heads, num_embeddings, embedding_dim, 1]
        )
        if add_to_values:
            embedding_size[-1] = 2
        initial_stddev = embedding_dim**(-0.5)
        self.embeddings = nn.Parameter(torch.zeros(*embedding_size))
        nn.init.normal_(self.embeddings, mean=0.0, std=initial_stddev)

    def forward(self, query, saved_state=None):
        """
        Computes relative positional embeddings to be added to keys (and optionally values),
        multiplies the embeddings for keys with queries to create positional logits,
        returns the positional logits, along with embeddings for values (optionally)
        which could be added to values outside this module.
        Args:
            query (torch.Tensor): query tensor
            saved_state (dict): saved state from previous time step
        Shapes:
            query: `(length, batch_size*num_heads, embed_dim)`
        Returns:
            tuple(torch.Tensor):
                - positional logits
                - relative positional embeddings to be added to values
        """
        # During inference when previous states are cached
        if saved_state is not None and "prev_key" in saved_state:
            assert not self.unmasked, "This should only be for decoder attention"
            length = saved_state["prev_key"].shape[-2] + 1  # `length - 1` keys are cached,
                                                            # `+ 1` for the current time step
            decoder_step = True
        else:
            length = query.shape[0]
            decoder_step = False

        used_embeddings = self.get_embeddings_for_query(length)

        values_embeddings = (
            used_embeddings[..., 1]
            if self.add_to_values
            else None
        )
        positional_logits = self.calculate_positional_logits(query, used_embeddings[..., 0])
        positional_logits = self.relative_to_absolute_indexing(positional_logits, decoder_step)
        return (positional_logits, values_embeddings)

    def get_embeddings_for_query(self, length):
        """
        Extract the required embeddings. The maximum relative position between two time steps is
        `length` for masked case or `2*length - 1` for the unmasked case. If `length` is greater than
        `max_relative_pos`, we first pad the embeddings tensor with zero-embeddings, which represent
        embeddings when relative position is greater than `max_relative_pos`. In case `length` is
        less than `max_relative_pos`, we don't use the first `max_relative_pos - length embeddings`.
        Args:
            length (int): length of the query
        Returns:
            torch.Tensor: embeddings used by the query
        """
        pad_length = max(length - self.max_relative_pos, 0)
        start_pos = max(self.max_relative_pos - length, 0)
        if self.unmasked:
            with torch.no_grad():
                padded_embeddings = nn.functional.pad(
                    self.embeddings,
                    (0, 0, 0, 0, pad_length, pad_length)
                )
            used_embeddings = padded_embeddings.narrow(-3, start_pos, 2*length - 1)
        else:
            with torch.no_grad():
                padded_embeddings = nn.functional.pad(
                    self.embeddings,
                    (0, 0, 0, 0, pad_length, 0)
                )
            used_embeddings = padded_embeddings.narrow(-3, start_pos, length)
        return used_embeddings

    def calculate_positional_logits(self, query, relative_embeddings):
        """
        Multiplies query with the relative positional embeddings to create relative
        positional logits
        Args:
            query (torch.Tensor): Input tensor representing queries
            relative_embeddings (torch.Tensor): relative embeddings compatible with query
        Shapes:
            query: `(length, batch_size*num_heads, embed_dim)` if heads share embeddings
                   else `(length, batch_size, num_heads, embed_dim)`
            relative_embeddings: `(max_allowed_relative_positions, embed_dim)` if heads share embeddings
                                 else `(num_heads, max_allowed_relative_positions, embed_dim)`
                                 where `max_allowed_relative_positions` is `length` if masked
                                 else `2*length - 1`
        Returns:
            torch.Tensor: relative positional logits
        """
        if self.heads_share_embeddings:
            positional_logits = torch.einsum("lbd,md->lbm", query, relative_embeddings)
        else:
            query = query.view(query.shape[0], -1, self.num_heads, self.embedding_dim)
            positional_logits = torch.einsum("lbhd,hmd->lbhm", query, relative_embeddings)
            positional_logits = positional_logits.contiguous().view(
                positional_logits.shape[0], -1, positional_logits.shape[-1]
            )
        # mask out tokens out of range
        length = query.size(0)
        if length > self.max_relative_pos:
            # there is some padding
            pad_length = length - self.max_relative_pos
            positional_logits[:,:,:pad_length] -= 1e8
            if self.unmasked:
                positional_logits[:,:,-pad_length:] -= 1e8
        return positional_logits

    def relative_to_absolute_indexing(self, x, decoder_step):
        """
        Index tensor x (relative positional logits) in terms of absolute positions
        rather than relative positions. Last dimension of x represents relative position
        with respect to the first dimension, whereas returned tensor has both the first
        and last dimension indexed with absolute positions.
        Args:
            x (torch.Tensor): positional logits indexed by relative positions
            decoder_step (bool): is this is a single decoder step (during inference)
        Shapes:
            x: `(length, batch_size*num_heads, length)` for masked case or
               `(length, batch_size*num_heads, 2*length - 1)` for unmasked
        Returns:
            torch.Tensor: positional logits represented using absolute positions
        """
        length, bsz_heads, _ = x.shape

        if decoder_step:
            return x.contiguous().view(bsz_heads, 1, -1)

        if self.unmasked:
            x = nn.functional.pad(
                x,
                (0, 1)
            )
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length * 2 * length)
            x = nn.functional.pad(
                x,
                (0, length - 1)
            )
            # Reshape and slice out the padded elements.
            x = x.view(bsz_heads, length + 1, 2*length - 1)
            return x[:, :length, length-1:]
        else:
            x = nn.functional.pad(
                x,
                (1, 0)
            )
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length+1, length)
            return x[:, 1:, :]
