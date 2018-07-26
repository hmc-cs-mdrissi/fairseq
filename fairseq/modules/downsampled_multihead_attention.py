# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.scalar_bias import scalar_bias

class SingleHeadAttention(nn.Module):
    """
    Single-head attention that supports Gating and Downsampling
    """
    def __init__(
        self, out_channels, embed_dim, head_dim, head_index, dropout=0.,
        bias=True, project_input=True, gated=False, downsample=False,
        num_heads=1, hierarchical_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.head_index = head_index
        self.head_dim = head_dim
        self.project_input = project_input
        self.gated = gated
        self.downsample = downsample
        self.num_heads = num_heads
        self.projection = None
        self.hierarchical_attention = hierarchical_attention

        k_layers = []
        v_layers = []
        if self.downsample:
            k_layers.append(Downsample(self.head_index))
            v_layers.append(Downsample(self.head_index))
            out_proj_size = self.head_dim
        else:
            out_proj_size = self.head_dim * self.num_heads
        
        if self.project_input:
            if self.gated:
                k_layers.append(GatedLinear(self.embed_dim, out_proj_size, bias=bias))
                self.in_proj_q = GatedLinear(self.embed_dim, out_proj_size, bias=bias)
                v_layers.append(GatedLinear(self.embed_dim, out_proj_size, bias=bias))
            else:
                k_layers.append(Linear(self.embed_dim, out_proj_size, bias=bias))
                self.in_proj_q = Linear(self.embed_dim, out_proj_size, bias=bias)
                v_layers.append(Linear(self.embed_dim, out_proj_size, bias=bias))
        elif self.gated:
            raise ValueError("You can't have both gated as True and project_input as False.")

        self.in_proj_k = nn.Sequential(*k_layers)
        self.in_proj_v = nn.Sequential(*v_layers)

        if self.downsample:
            self.out_proj = Linear(out_proj_size, self.head_dim, bias=bias)
        else:
            self.out_proj = Linear(out_proj_size, out_channels, bias=bias)

        if self.hierarchical_attention:
            if self.project_input:
                if self.gated:
                    self.in_proj_v_sentence = GatedLinear(self.embed_dim, out_proj_size, bias=bias)
                    self.in_proj_q_sentence = GatedLinear(self.embed_dim, out_proj_size, bias=bias)
                else:
                    self.in_proj_v_sentence = Linear(self.embed_dim, out_proj_size, bias=bias)
                    self.in_proj_q_sentence = Linear(self.embed_dim, out_proj_size, bias=bias)

        self.scaling = self.head_dim**-0.5

    def forward(
        self, query, key, value, sentence_value=None, all_chunk_sizes=None, 
        mask_future_timesteps=False, key_padding_mask=None, use_scalar_bias=False,
    ):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        src_len, bsz, out_channels = key.size()
        tgt_len = query.size(0)

        assert list(query.size()) == [tgt_len, bsz, out_channels]
        assert key.size() == value.size()

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.downsample:
            size = bsz
        else:
            size = bsz * self.num_heads

        k = key
        v = value
        q = query
        if self.project_input:
            q = self.in_proj_q(q)
            k = self.in_proj_k(k)
            v = self.in_proj_v(v)
            src_len = k.size()[0]

        q *= self.scaling

        if self.hierarchical_attention:
            v_sentence = sentence_value
            q_sentence = query
            if self.project_input:
                # B is used for batch, S for number of sentences, T for source length (reminder this is affected by downsampling), T' for target length,
                # and C for channels.
                v_sentence = self.in_proj_v_sentence(v_sentence) # B x S x C
                q_sentence = self.in_proj_q_sentence(q_sentence) # T' x B x C

            q_sentence *= self.scaling

        if not self.downsample:
            q = q.view(tgt_len, size, self.head_dim)
            k = k.view(src_len, size, self.head_dim)
            v = v.view(src_len, size, self.head_dim)

        q = q.transpose(0, 1) # B x T' x C
        k = k.transpose(0, 1) # B x T  x C
        v = v.transpose(0, 1) # B x T  x C

        attn_weights = torch.bmm(q, k.transpose(1, 2)) # B x T' x T

        if self.hierarchical_attention:
            q_sentence = q_sentence.transpose(0, 1) # B x T' x C
            sentence_attn_weights = torch.bmm(q_sentence, v_sentence.transpose(1, 2)) # B x T' x S
            sentence_attn_weights = F.softmax(sentence_attn_weights, dim=-1)
        else:
            sentence_attn_weights = None

        if mask_future_timesteps:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights *= torch.tril(
                attn_weights.data.new([1]).expand(tgt_len, tgt_len).clone(),
                diagonal=-1,
            )[:, ::self.head_index + 1 if self.downsample else 1].unsqueeze(0)
            attn_weights += torch.triu(
                attn_weights.data.new([-math.inf]).expand(tgt_len, tgt_len).clone(),
                diagonal=0
            )[:, ::self.head_index + 1 if self.downsample else 1].unsqueeze(0)

        tgt_size = tgt_len
        if use_scalar_bias:
            attn_weights = scalar_bias(attn_weights, 2)
            v = scalar_bias(v, 1)
            tgt_size += 1

        if key_padding_mask is not None:
            # don't attend to padding symbols
            if key_padding_mask.max() > 0:
                if self.downsample:
                    attn_weights = attn_weights.view(bsz, 1, tgt_len, src_len)
                else:
                    attn_weights = attn_weights.view(size, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -math.inf,
                )
                attn_weights = attn_weights.view(size, tgt_len, src_len)
        
        if self.hierarchical_attention:
            new_attn_weights = []
            # attn_weight: T' x T, sentence_attn_weight: T' x S, chunk_sizes: S
            for attn_weight, sentence_attn_weight, chunk_sizes in zip(attn_weights, sentence_attn_weights, all_chunk_sizes):
                new_attn_weights.append(torch.cat(list(map(lambda word_weights, sentence_weight: F.softmax(word_weights, dim=-1) * sentence_weight.unsqueeze(1),  
                                                           attn_weight.split(chunk_sizes.tolist(), dim=1), 
                                                           sentence_attn_weight.transpose(0,1))), dim=1))
            attn_weights = torch.stack(new_attn_weights)
        else:
            attn_weights = F.softmax(attn_weights, dim=-1)

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training) # B x T x T

        attn = torch.bmm(attn_weights, v)

        if self.downsample:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.head_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)

        attn = self.out_proj(attn)

        return {'attn': attn, 'attn_weights': attn_weights, 'sentence_attn_weights': sentence_attn_weights}

class DownsampledMultiHeadAttention(nn.ModuleList):
    """
    Multi-headed attention with Gating and Downsampling
    """
    def __init__(
        self, out_channels, embed_dim, num_heads, dropout=0., bias=True,
        project_input=True, gated=False, downsample=False, hierarchical_attention=False,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.downsample = downsample
        self.gated = gated
        self.project_input = project_input
        self.hierarchical_attention = hierarchical_attention
        assert self.head_dim * num_heads == embed_dim

        if self.downsample:
            attention_heads = []
            for index in range(self.num_heads):
                attention_heads.append(
                    SingleHeadAttention(
                        out_channels, self.embed_dim, self.head_dim, index,
                        self.dropout, bias, self.project_input, self.gated,
                        self.downsample, self.num_heads, hierarchical_attention=hierarchical_attention
                    )
                )
            super().__init__(modules=attention_heads)
            self.out_proj = Linear(embed_dim, out_channels, bias=bias)
        else:
            # either we have a list of attention heads, or just one attention head
            # if not being downsampled, we can do the heads with one linear layer instead of separate ones
            super().__init__()
            self.attention_module = SingleHeadAttention(
                out_channels, self.embed_dim, self.head_dim, 1, self.dropout,
                bias, self.project_input, self.gated, self.downsample, self.num_heads, 
                hierarchical_attention=hierarchical_attention
            )

    def forward(
        self, query, key, value, sentence_value=None, all_chunk_sizes=None, 
        mask_future_timesteps=False, key_padding_mask=None, use_scalar_bias=False,
    ):
        src_len, bsz, embed_dim = key.size()
        tgt_len = query.size(0)
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if self.hierarchical_attention and (sentence_value is None or all_chunk_sizes is None):
            raise ValueError("You can't have hierarchical attention without passing in non-None values for sentence value"
                             " and all chunk sizes.")

        tgt_size = tgt_len
        if use_scalar_bias:
            tgt_size += 1

        attn = []
        attn_weights = []
        sentence_attn_weights = []
        if self.downsample:
            for attention_head_number in range(self.num_heads):
                # call the forward of each attention head
                attention_output = self[attention_head_number](
                    query, key, value, sentence_value=sentence_value, all_chunk_sizes=all_chunk_sizes, 
                    mask_future_timesteps=mask_future_timesteps, key_padding_mask=key_padding_mask, 
                    use_scalar_bias=use_scalar_bias,
                )
                attn.append(attention_output['attn'])
                attn_weights.append(attention_output['attn_weights'])
                sentence_attn_weights.append(attention_output['sentence_attn_weights'])

            full_attn = torch.cat(attn, dim=2)
            full_attn = self.out_proj(full_attn)

            if self.hierarchical_attention:
                return full_attn, attn_weights[0].clone(), sentence_attn_weights[0].clone()
            else:
                return full_attn, attn_weights[0].clone(), sentence_attn_weights[0]
        else:
            attention_output = self.attention_module(
                query, key, value, sentence_value=sentence_value, all_chunk_sizes=all_chunk_sizes,
                mask_future_timesteps=mask_future_timesteps, key_padding_mask=key_padding_mask, 
                use_scalar_bias=use_scalar_bias,
            )

            full_attn_weights = attention_output['attn_weights']
            full_attn_weights = full_attn_weights.view(bsz, self.num_heads, tgt_size, src_len)
            full_attn_weights = full_attn_weights.mean(dim=1)

            sentence_attn_weights = attention_output['sentence_attn_weights']
            if self.hierarchical_attention:
                sentence_attn_weights = sentence_attn_weights.view(bsz, self.num_heads, tgt_size, -1)
                sentence_attn_weights = sentence_attn_weights.mean(dim=1) 
            return attention_output['attn'], full_attn_weights, sentence_attn_weights


class Downsample(nn.Module):
    """
    Selects every nth element, where n is the index
    """
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[::self.index+1]


def Linear(in_features, out_features, dropout=0., bias=True):
    """Weight-normalized Linear layer (input: B x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def GatedLinear(in_features, out_features, dropout=0., bias=True):
    """Weight-normalized Linear layer (input: B x T x C) with interspersed GLU units"""
    return nn.Sequential(
        Linear(in_features, out_features*4, dropout, bias),
        nn.GLU(),
        Linear(out_features*2, out_features*2, dropout, bias),
        nn.GLU(),
        Linear(out_features, out_features, dropout, bias)
    )
