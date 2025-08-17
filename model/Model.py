"""Code."""
import os
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import HieraAttLayer, ResLevelPool, get_features
from utils.logger import Logger

logger = Logger.logger


class IdEmbedder(nn.Module):
    """Define Class IdEmbedder."""

    def __init__(self, config):
        """Run __init__ method."""
        # code.
        super(IdEmbedder, self).__init__()
        self.id_bn = config['id_bn']
        if self.id_bn:
            self.batch_norm = torch.nn.BatchNorm1d(config['id']['N0'])
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(config['id']['N0'], config['id']['N1']),
            torch.nn.ELU(),
            torch.nn.Linear(config['id']['N1'], config['id']['N2']),
        )  # 59 --> 64 --> 32

    def forward(self, id_f):
        """Run forward method."""
        # code.
        if self.id_bn:
            id_f = self.batch_norm(id_f)
        id_f = self.transform(id_f)
        return id_f


class CrossAttention(nn.Module):
    """Define Class CrossAttention."""

    # TODO: use flash attention
    def __init__(self, dim=16, n_layer=4, num_heads=2, dropout=0.01, seed=None):
        """Run __init__ method."""
        # code.
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.n_layer = n_layer

        self.layer_norm_q = nn.LayerNorm(dim)
        self.layer_norm_k = nn.LayerNorm(dim)
        self.layer_norm_v = nn.LayerNorm(dim)

        linears = []
        for i in range(3):
            linear = nn.Linear(dim, dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            linears.append(linear)

        self.query_projection = linears[0]
        self.key_projection = linears[1]
        self.value_projection = linears[2]

        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(dim, dim)
        self.att_module = DmasifAttentionModule(dim, seed=seed)

    def forward(self, query, key, value, w, mask):
        """Run forward method."""
        # code.
        batch_size = query.size(0)

        # Project inputs for all heads
        d_by_h = self.dim // self.num_heads

        query = self.layer_norm_q(query)
        key = self.layer_norm_k(key)
        value = self.layer_norm_v(value)
        query0 = query

        # Q, K, V: [batch_size, num_heads, seq_len, d_by_h]
        Q = self.query_projection(query).view(batch_size, -1, self.num_heads, d_by_h).transpose(1, 2)
        K = self.key_projection(key).view(batch_size, -1, self.num_heads, d_by_h).transpose(1, 2)
        V = self.value_projection(value).view(batch_size, -1, self.num_heads, d_by_h).transpose(1, 2)

        # Compute scaled dot-product self-attention
        Q = Q / d_by_h ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores + w

        # remove max
        scores = scores - scores.max(dim=-1, keepdim=True)[0]

        scores = scores + mask
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, d_by_h]

        # Concatenate all heads output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        output = self.output_projection(context)

        output = output + query0
        return output, attn


class CrossNet(CrossAttention):
    """Define Class CrossNet."""

    def cross(self, emd1, emd2, dists, mask):
        """Run cross method."""
        # code.
        query = emd1  # [N_lig, d]
        key = emd2  # [N_rec, d]
        value = emd2  # [N_rec, d]
        output1, attn1 = super().forward(query, key, value, dists[:, None, ...], mask[:, None, ...])  # [1, N_lig, d], [1, N_lig, N_rec]

        # lig to rec
        query = emd2  # [N_rec, d]
        key = emd1  # [N_lig, d]
        value = emd1  # [N_lig, d]
        output2, attn2 = super().forward(
            query, key, value, torch.transpose(dists, -1, -2)[:, None, ...], torch.transpose(mask, -1, -2)[:, None, ...]
        )  # [1, N_rec, d], [1, N_rec, N_lig]
        return output1, output2

    def forward(self, emd1, emd2, dists, mask, flat_mask):
        """Run forward method."""
        # code.
        for _ in range(self.n_layer):
            emd1, emd2 = self.cross(emd1, emd2, dists, mask)

        # update
        out = torch.cat([emd1, emd2], 1)
        out = nn.functional.layer_norm(out, [16])    # [N_patch1 + N_patch2, 16]
        out = self.att_module(out, flat_mask)
        return out


class DmasifAttentionModule(nn.Module):
    """Define Class DmasifAttentionModule."""

    def __init__(self, dim, seed=None):
        """Run __init__ method."""
        # code.
        super(DmasifAttentionModule, self).__init__()
        self.fc = nn.Linear(dim, dim)
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            self.attention_vector = nn.Parameter(torch.rand(dim, generator=g))
        else:
            self.attention_vector = nn.Parameter(torch.rand(dim, generator=None))

    def forward(self, x, flat_mask):
        """Run forward method."""
        # code.
        output = []
        for _ in range(x.shape[0]):
            flat_mask_i = flat_mask[_] == 1
            x_i = x[_][flat_mask_i]    # (N_res, 16)
            attention_weights = self.fc(x_i)    # (N_res, 16)
            attention_weights = torch.matmul(attention_weights, self.attention_vector)    # (N_res)
            softmax_attention_weights = F.softmax(attention_weights, dim=0)    # (N_res)

            output.append(torch.matmul(softmax_attention_weights, x_i))
        output = torch.stack(output)
        return output


class ResLevelModule(torch.nn.Module):
    """Define Class ResLevelModule."""

    def __init__(self, config):
        """Run __init__ method."""
        # code.
        super(ResLevelModule, self).__init__()
        self.config = config
        self.fr_bn = config['fr_bn']
        if self.fr_bn:
            self.batch_norm = torch.nn.BatchNorm1d(2 * config['dm']['N0'])
        self.nf = torch.nn.Sequential(
            torch.nn.Linear(config['nf']['N0'], config['nf']['N1']),
            torch.nn.ELU(),
            torch.nn.Linear(config['nf']['N1'], config['nf']['N1']),
            torch.nn.ELU(),
            torch.nn.Linear(config['nf']['N1'], config['nf']['N1']),
        )  # 30 --> 32 --> 32 --> 32

        nn = [int(x) for x in config['hat']['nn'].split(',')]
        hat_config = [{'Ns': config['hat']['Ns'],
                       'Nh': config['hat']['Nh'],
                       'Nk': config['hat']['Nk'],
                       'nn': x} for x in nn for _ in range(config['hat']['n_layer_per_block'])]
        self.hat = torch.nn.Sequential(*[HieraAttLayer(layer_params) for layer_params in hat_config])
        self.rlp = ResLevelPool(config['rlp']['N0'], config['rlp']['N1'], config['rlp']['Nh'])

        self.dm = torch.nn.Sequential(
            torch.nn.Linear(2 * config['dm']['N0'] + config['id']['N2'], config['dm']['N0_']),
            torch.nn.ELU(),
            torch.nn.Linear(config['dm']['N0_'], config['dm']['N1']),
            torch.nn.ELU(),
            torch.nn.Linear(config['dm']['N1'], config['dm']['N1']),
            torch.nn.ELU(),
            torch.nn.Linear(config['dm']['N1'], config['dm']['N2']),
        )  # 96 --> 64 --> 32 --> 32 --> 16

    def forward(self, X, ids_topk, ft0, M, id_f):
        """Run forward method."""
        # code.
        # X: [N_atom_batch, 3], ids_topk: [N_atom_batch, 64], ft0: [N_atom_batch, 30], M: [N_atom_batch, N_res]
        ft_0 = self.nf.forward(ft0)  # [N_atom_batch, 32]

        ft_1 = torch.zeros((ft_0.shape[0] + 1, X.shape[1], ft_0.shape[1]), device=X.device)  # [N_atom_batch + 1, 3, 32]
        # ft_0: [N_atom_batch + 1, 32], ids_topk: [N_atom_batch + 1, 64], D_nn: [N_atom_batch + 1, 64], R_nn: [N_atom_batch + 1, 64, 3]
        ft_0, ids_topk, D_nn, R_nn = get_features(X, ids_topk, ft_0)

        # ft_0a: [N_atom_batch + 1, 32], ft_1a: [N_atom_batch + 1, 3, 32]
        ft_0a, ft_1a, _, _, _ = self.hat.forward((ft_0, ft_1, ids_topk, D_nn, R_nn))
        # ft_0r: [N_res, 32], ft_1r: [N_res, 3, 32]
        ft_0r, ft_1r = self.rlp.forward(ft_0a[1:], ft_1a[1:], M)

        fr = torch.cat([ft_0r, torch.norm(ft_1r, dim=1)], dim=1)  # [N_res, 64]
        if self.fr_bn:
            fr = self.batch_norm(fr)

        fr = torch.cat([fr, id_f], dim=1)
        f = self.dm.forward(fr)  # [N_res, 16]

        return f


class AttentionModule(nn.Module):
    """Define Class AttentionModule."""

    def __init__(self, config, seed=None):
        """Run __init__ method."""
        # code.
        super(AttentionModule, self).__init__()
        self.fc = nn.Linear(config['dm']['N2'], config['att']['N0'])
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            self.attention_vector = nn.Parameter(torch.rand(config['att']['N0'], generator=g))
        else:
            self.attention_vector = nn.Parameter(torch.rand(config['att']['N0'], generator=None))

    def forward(self, x, residue_mask):
        """Run forward method."""
        # code.
        b = int(residue_mask.max().item()) + 1
        n = x.size(0)

        mask = torch.zeros(n, b, device=x.device)
        mask[torch.arange(n), residue_mask] = 1

        attention_weights = self.fc(x)    # (N_res, 16)
        attention_weights = torch.matmul(attention_weights, self.attention_vector)    # (N_res)
        softmax_attention_weights = torch.zeros(n, b, device=x.device)    # (N_res, b)

        for i in range(b):
            mask = (residue_mask == i)
            softmax_attention_weights[mask, i] = F.softmax(attention_weights[mask], dim=0)
        output = torch.matmul(softmax_attention_weights.transpose(0, 1), x)  # [b, m]
        return output


class ScoreModel(nn.Module):
    """Define Class ScoreModel."""

    def __init__(self, config, seed=None):
        """Run __init__ method."""
        # code.
        super(ScoreModel, self).__init__()
        self.config = config

        self.id_embedder = IdEmbedder(config)
        self.feature_module = ResLevelModule(config)
        self.attention_module = AttentionModule(config, seed=seed)
        self.fc1 = nn.Linear(config['dm']['N2'], config['fc1']['N1'])
        self.fc2 = nn.Linear(config['fc2']['N0'], config['fc2']['N1'])

        last_layer_bias = self.config.get('last_layer_bias', True)
        self.fc3 = nn.Linear(config['fc2']['N1'], config['out'], bias=last_layer_bias)

        self.sigmoid_index = config.get('sigmoid_index', range(config['out']))
    def forward(self, batch):
        """Run forward method."""
        # code.
        id_f = self.id_embedder(batch["id_f"])
        # X: [N_atom_batch, 3], ids_topk: [N_atom_batch, 64], ft0: [N_atom_batch, 30], M: [N_atom_batch, N_res], residue_mask: [N_res]
        fe = self.feature_module(batch["X"], batch["ids_topk"], batch["q"], batch["M"], id_f)  # [N_res, 16]
        x = self.attention_module(fe, batch["residue_mask"])  # [B, 16]

        x = F.relu(self.fc1(x))  # [B, 16]
        x = F.relu(self.fc2(x))  # [B, 8]
        x = self.fc3(x)          # [B, out]

        if self.sigmoid_index:
            sigmoid_out = torch.sigmoid(x[:,self.sigmoid_index])
            final_out = x.clone()
            final_out[:,self.sigmoid_index] = sigmoid_out
            score = final_out
        else:
            score = x

        return score


class ScoreModelv2(nn.Module):
    """Define Class ScoreModelv2."""

    def __init__(self, config, seed=None):
        """Run __init__ method."""
        # code.
        super(ScoreModelv2, self).__init__()
        self.config = config

        self.id_embedder = IdEmbedder(config)
        self.feature_module = ResLevelModule(config)
        self.attention_module = AttentionModule(config, seed=seed)
        self.fc1 = nn.Linear(config['dm']['N2'], config['fc1']['N1'])
        self.fc2 = nn.Linear(config['fc2']['N0'], config['fc2']['N1'])

        last_layer_bias = self.config.get('last_layer_bias', True)
        self.fc3 = nn.Linear(config['fc2']['N1'], config['out'], bias=last_layer_bias)

        self.sigmoid_index = config.get('sigmoid_index', range(config['out']))
    def forward(self, batch):
        """Run forward method."""
        # code.
        id_f = self.id_embedder(batch["id_f"])
        # X: [N_atom_batch, 3], ids_topk: [N_atom_batch, 64], ft0: [N_atom_batch, 30], M: [N_atom_batch, N_res], residue_mask: [N_res]
        fe = self.feature_module(batch["X"], batch["ids_topk"], batch["q"], batch["M"], id_f)  # [N_res, 16]
        x = self.attention_module(fe, batch["residue_mask"])  # [B, 16]

        x = F.relu(self.fc1(x))  # [B, 16]
        x = F.relu(self.fc2(x))  # [B, 8]
        x = self.fc3(x)          # [B, out]

        if self.sigmoid_index:
            sigmoid_out = torch.sigmoid(x[:,self.sigmoid_index])
            final_out = x.clone()
            final_out[:,self.sigmoid_index] = sigmoid_out
            score = final_out
        else:
            score = x

        return score
