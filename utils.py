'''
utility functions
'''
import numpy as np
import torch
import torch.nn as nn
import os
import pdb
import torch.nn.functional as F

def normalize_duration(input, mask):
    input = torch.exp(input)*mask
    output = F.normalize(input, p=1, dim=-1)
    return output

def read_mapping_dict(file_path):
    # github.com/yabufarha/anticipating-activities
    '''This function read action index from the txt file'''
    file_ptr = open(file_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

def eval_file(gt_content, recog_content, obs_percentage, classes):
    # github.com/yabufarha/anticipating-activities
    last_frame = min(len(recog_content), len(gt_content))
    recognized = recog_content[int(obs_percentage * len(gt_content)):last_frame]
    ground_truth = gt_content[int(obs_percentage * len(gt_content)):last_frame]

    n_T = np.zeros(len(classes))
    n_F = np.zeros(len(classes))
    for i in range(len(ground_truth)):
        if ground_truth[i] == recognized[i]:
            n_T[classes[ground_truth[i]]] += 1
        else:
            n_F[classes[ground_truth[i]]] += 1

    return n_T, n_F

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''Apply label smoothing if needed'''

    loss = cal_loss(pred, gold.long(), trg_pad_idx, smoothing=smoothing)
    pred = pred.max(1)[1]
#    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''Calculate cross entropy loss, apply label smoothing if needed'''

#    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1) + 1
        B = pred.size(0)

        one_hot = torch.zeros((B, n_class)).to(pred.device).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class -1)
        one_hot = one_hot[:, :-1]
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
        loss = loss / non_pad_mask.sum()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx)
    return loss




