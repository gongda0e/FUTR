import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import pdb
from einops import repeat, rearrange
from model.extras.transformer import Transformer
from model.extras.position import PositionalEncoding

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

class FUTR(nn.Module):

    def __init__(self, n_class, hidden_dim, src_pad_idx, device, args, n_query=8, n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.device = device
        self.hidden_dim = hidden_dim
        self.input_embed = nn.Linear(args.input_dim, hidden_dim)
        self.transformer = Transformer(hidden_dim, n_head, num_encoder_layers, num_decoder_layers,
                                        hidden_dim*4, normalize_before=False)
        self.n_query = n_query
        self.args = args
        nn.init.xavier_uniform_(self.input_embed.weight)
        self.query_embed = nn.Embedding(self.n_query, hidden_dim)


        if args.seg :
            self.fc_seg = nn.Linear(hidden_dim, n_class-1) #except SOS, EOS
            nn.init.xavier_uniform_(self.fc_seg.weight)

        if args.anticipate :
            self.fc = nn.Linear(hidden_dim, n_class)
            nn.init.xavier_uniform_(self.fc.weight)
            self.fc_len = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(self.fc_len.weight)

        if args.pos_emb:
            #pos embedding
            max_seq_len = args.max_pos_len
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
            nn.init.xavier_uniform_(self.pos_embedding)
            # Sinusoidal position encoding
            self.pos_enc = PositionalEncoding(hidden_dim)

        if args.input_type =='gt':
            self.gt_emb = nn.Embedding(n_class+2, self.hidden_dim, padding_idx=n_class+1)
            nn.init.xavier_uniform_(self.gt_emb.weight)

    def forward(self, inputs, mode='train'):
        if mode == 'train' :
            src, src_label = inputs
            tgt_key_padding_mask = None
            src_key_padding_mask = get_pad_mask(src_label, self.src_pad_idx).to(self.device)
            memory_key_padding_mask = src_key_padding_mask.clone().to(self.device)
        else :
            src = inputs
            src_key_padding_mask = None
            memory_key_padding_mask = None
            tgt_key_padding_mask = None

        tgt_mask = None

        if self.args.input_type == 'i3d_transcript':
            B, S, C = src.size()
            src = self.input_embed(src) #[B, S, C]
        elif self.args.input_type == 'gt':
            B, S = src.size()
            src = self.gt_emb(src)
        src = F.relu(src)

        # action query embedding
        action_query = self.query_embed.weight
        action_query = action_query.unsqueeze(0).repeat(B, 1, 1)
        tgt = torch.zeros_like(action_query)

        # pos embedding
        pos = self.pos_embedding[:, :S,].repeat(B, 1, 1)
        src = rearrange(src, 'b t c -> t b c')
        tgt = rearrange(tgt, 'b t c -> t b c')
        pos = rearrange(pos, 'b t c -> t b c')
        action_query = rearrange(action_query, 'b t c -> t b c')

        src, tgt = self.transformer(src, tgt, src_key_padding_mask, tgt_mask, None, action_query, pos, None)

        tgt = rearrange(tgt, 't b c -> b t c')
        src = rearrange(src, 't b c -> b t c')

        output = dict()
        if self.args.anticipate :
            # action anticipation
            output_class = self.fc(tgt) #[T, B, C]
            duration = self.fc_len(tgt) #[B, T, 1]
            duration = duration.squeeze(2) #[B, T]
            output['duration'] = duration
            output['action'] = output_class

        if self.args.seg :
            # action segmentation
            tgt_seg = self.fc_seg(src)
            output['seg'] = tgt_seg

        return output


def get_pad_mask(seq, pad_idx):
    return (seq ==pad_idx)



