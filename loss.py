import torch
import torch.nn.functional as F
import numpy as np

from data_utils import phoneme_inventory
from utils import align_from_distances

def dtw_loss(predictions, phoneme_predictions, speech_features, phoneme_targets, silents, phoneme_loss_weight, target_lengths, est_lengths, phoneme_eval=False, phoneme_confusion=None, dimension_norm=False): 
    if phoneme_confusion is None:
        phoneme_confusion = np.zeros((len(phoneme_inventory),len(phoneme_inventory)))
    # NOTE:
    # - Exclude 'spn' (spoken noise) targets from loss/accuracy aggregation.
    # - If 'spn' is not in the inventory, behavior is unchanged.
    try:
        spn_id = phoneme_inventory.index("spn")
    except ValueError:
        spn_id = None
    device = predictions.device
    # Feature-dimension normalization factor
    if not dimension_norm:
        D_global = torch.tensor(1.0, device=device, dtype=predictions.dtype)
    else:
        D_global = torch.sqrt(torch.tensor([predictions.shape[-1]], dtype=torch.float32, device=device)).to(dtype=predictions.dtype).squeeze(0)
    losses = []
    losses_dist = []
    losses_ph = []
    correct_phones = 0
    total_length = 0
    total_ph_length = 0
    for pred, y, pred_phone, y_phone, silent, target_length, est_length in zip(predictions, speech_features, phoneme_predictions, phoneme_targets, silents, target_lengths, est_lengths):
        assert len(pred.size()) == 2 and len(y.size()) == 2
        pred = pred[:est_length] if silent else pred[:target_length]
        y = y[:target_length]
        pred_phone = pred_phone[:est_length] if silent else pred_phone[:target_length]
        y_phone = y_phone[:target_length]
        D = D_global
        y_phone = y_phone.to(device)
        if spn_id is None:
            valid_phone_mask = None
        else:
            valid_phone_mask = y_phone != int(spn_id)
            total_ph_length += int(valid_phone_mask.sum().item())

        if silent:
            dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0))
            dists = dists.squeeze(0)/D

            # pred_phone (seq1_len, 48), y_phone (seq2_len)
            # phone_probs (seq1_len, seq2_len)
            pred_phone = F.log_softmax(pred_phone, -1)
            phone_lprobs = pred_phone[:,y_phone]

            if valid_phone_mask is None:
                costs = dists*0.5 + phoneme_loss_weight * -phone_lprobs / D
            else:
                # Set phoneme cost to 0 for spn-target columns so they do not affect alignment/loss.
                phon_cost = phoneme_loss_weight * -phone_lprobs / D
                if (~valid_phone_mask).any():
                    phon_cost[:, ~valid_phone_mask] = 0
                costs = dists*0.5 + phon_cost

            alignment = align_from_distances(costs.T.cpu().detach().numpy())

            loss = costs[alignment,range(len(alignment))].sum()
            loss_dist = dists[alignment,range(len(alignment))].sum()
            if valid_phone_mask is None:
                loss_ph = -phone_lprobs[alignment,range(len(alignment))].sum()
            else:
                if valid_phone_mask.any():
                    idx = torch.nonzero(valid_phone_mask, as_tuple=False).squeeze(-1)
                    ali = torch.as_tensor(alignment, device=device, dtype=torch.long)[idx]
                    loss_ph = -phone_lprobs[ali, idx].sum()
                else:
                    loss_ph = torch.zeros((), device=device, dtype=phone_lprobs.dtype)

            if phoneme_eval:
                alignment = align_from_distances(costs.T.cpu().detach().numpy())

                pred_phone = pred_phone.argmax(-1)
                if valid_phone_mask is None:
                    correct_phones += (pred_phone[alignment] == y_phone).sum().item()
                else:
                    if valid_phone_mask.any():
                        idx = torch.nonzero(valid_phone_mask, as_tuple=False).squeeze(-1)
                        ali = torch.as_tensor(alignment, device=device, dtype=torch.long)[idx]
                        correct_phones += (pred_phone[ali] == y_phone[idx]).sum().item()

                if valid_phone_mask is None:
                    for p, t in zip(pred_phone[alignment].tolist(), y_phone.tolist()):
                        phoneme_confusion[p, t] += 1
                else:
                    if valid_phone_mask.any():
                        idx = torch.nonzero(valid_phone_mask, as_tuple=False).squeeze(-1)
                        ali_cpu = np.asarray(alignment, dtype=np.int64)[idx.detach().cpu().numpy()]
                        for p, t in zip(pred_phone.detach().cpu().numpy()[ali_cpu].tolist(), y_phone.detach().cpu().numpy()[idx.detach().cpu().numpy()].tolist()):
                            phoneme_confusion[p, t] += 1
        else:
            assert y.size(0) == pred.size(0)

            dists = F.pairwise_distance(y, pred)/D
            loss_dist = dists.sum()

            assert len(pred_phone.size()) == 2 and len(y_phone.size()) == 1
            if spn_id is None:
                loss_ph = F.cross_entropy(pred_phone, y_phone, reduction='sum')
            else:
                loss_ph = F.cross_entropy(pred_phone, y_phone, reduction='sum', ignore_index=int(spn_id))
            loss = loss_dist*0.5 + phoneme_loss_weight * loss_ph / D 

            if phoneme_eval:
                pred_phone = pred_phone.argmax(-1)
                if valid_phone_mask is None:
                    correct_phones += (pred_phone == y_phone).sum().item()
                else:
                    if valid_phone_mask.any():
                        correct_phones += ((pred_phone == y_phone) & valid_phone_mask).sum().item()

                if valid_phone_mask is None:
                    for p, t in zip(pred_phone.tolist(), y_phone.tolist()):
                        phoneme_confusion[p, t] += 1
                else:
                    if valid_phone_mask.any():
                        for p, t in zip(pred_phone[valid_phone_mask].tolist(), y_phone[valid_phone_mask].tolist()):
                            phoneme_confusion[p, t] += 1

        losses.append(loss)
        losses_dist.append(loss_dist)
        losses_ph.append(loss_ph)
        total_length += y.size(0)
    if spn_id is None:
        # Default behavior (normalize by total length)
        L = sum(losses)/total_length
        L_dist = sum(losses_dist)/total_length
        L_ph = sum(losses_ph)/total_length
        acc = correct_phones/total_length
    else:
        # dist uses all frames; phoneme/acc exclude spn frames.
        L_dist = sum(losses_dist)/total_length
        if total_ph_length > 0:
            L_ph = sum(losses_ph)/total_ph_length
            acc = correct_phones/total_ph_length
        else:
            L_ph = torch.zeros((), device=device, dtype=predictions.dtype)
            acc = 0.0
        # Combine terms normalized by their respective denominators.
        L = L_dist*0.5 + phoneme_loss_weight * L_ph / D_global

    return L, L_dist, L_ph, acc
