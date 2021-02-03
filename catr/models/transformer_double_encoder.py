# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, cnn_encoder_type=2,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        
        # 2 encoder blocks
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder2 = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        
        self.encoder_projection = nn.Conv2d(
            d_model*2, d_model, kernel_size=1)
        
        # Double width decoder layers
        self.embeddings = DecoderEmbeddings(config)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src2, mask, mask2, pos_embed, pos_embed2, tgt, tgt_mask):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        
        # adding another encoder object for inputting another cnn representation of the Xray
        src2 = src2.flatten(2).permute(2, 0, 1)
        pos_embed2 = pos_embed2.flatten(2).permute(2, 0, 1)
        mask2 = mask2.flatten(1)
        
        tgt = self.embeddings(tgt).permute(1, 0, 2)
        query_embed = self.embeddings.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)
        
        memory1 = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        memory2 = self.encoder2(src2, src_key_padding_mask=mask2, pos=pos_embed2)
        
#         memory = self.encoder_projection(torch.cat((memory1, memory2), dim=2))
        
        
#         print("src shape", src.shape)
#         print("memory shape", memory1.shape)
#         print("pos shape", pos_embed.shape)
#         print("is identical? ", pos_embed[0]==pos_embed[1])
#         print("mask shape", mask.shape)
#         concat_memory = torch.cat((memory, memory2), dim=2)
#         concat_pos = torch.cat((pos_embed, pos_embed2), dim=2)
#         concat_mask = torch.cat((mask, mask2), dim=1) # TODO: shouldn't this be 0?
        
        # Should use same memory mask, since it doesn't depend on entry dimension
        concat_mask = mask
        
#         print("memory concat shape", concat_memory.shape)
#         print("pos concat shape", concat_pos.shape)
#         print("mask concat shape", concat_mask.shape)
        
#         print("tgt shape", tgt.shape)
#         print("tgt mask shape", tgt_mask.shape)
#         print("queyr embed shape", query_embed.shape)
        
#         double_tgt = torch.cat((tgt, tgt), dim=2)
#         double_query = torch.cat((query_embed, query_embed), dim=2)
        
#         print("")
        
        hs = self.decoder(tgt, memory1, memory2,
                          memory_key_padding_mask=concat_mask, 
                          tgt_key_padding_mask=tgt_mask,
                          pos1=pos_embed, pos2=pos_embed2,
                          query_pos=query_embed,
                          tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))

        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory1, memory2,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos1: Optional[Tensor] = None,
                pos2: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for i, layer in enumerate(self.layers):
            output = layer(output, memory1, memory2, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos1=pos1, pos2=pos2, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        self.middle_position_embeddings = nn.Embedding(
#             config.max_position_embeddings, config.hidden_dim
            128, d_model
        )

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask.to("cuda"),
                              key_padding_mask=tgt_key_padding_mask.to("cuda"))[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory1, memory2,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos1: Optional[Tensor] = None,
                    pos2: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):

        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask.to("cuda"),
                              key_padding_mask=tgt_key_padding_mask.to("cuda"))[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        
#         print("query: ", tgt2.shape)
#         print("key: ", memory.shape)
#         print("value: ", memory.shape)
#         print("query_pos: ", query_pos.shape)
        
#         query = torch.cat((query,query), dim=2)
        
#         print("REAL query: ", query.shape)
#         print("key padding mask size:", memory_key_padding_mask.shape)
        
        
        
#         tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
        tgt2 = self.multihead_attn1(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory1, pos1),
                                   value=memory1, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt3 = self.norm3(tgt)
        
        bs = tgt3.shape[1]
#         position_ids = torch.arange(
#             seq_length, dtype=torch.long)
#         position_ids = position_ids.unsqueeze(0).expand(tgt3.shape)
#         middle_embeds = self.middle_position_embeddings(position_ids)
        middle_embeds = self.middle_position_embeddings.weight.unsqueeze(1)
        middle_embeds = middle_embeds.repeat(1, bs, 1)
        
        tgt3 = self.multihead_attn2(query=self.with_pos_embed(tgt3, middle_embeds),
                                   key=self.with_pos_embed(memory2, pos2),
                                   value=memory2, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout3(tgt3)
        tgt3 = self.norm4(tgt)
        
        tgt3 = self.linear2(self.dropout(self.activation(self.linear1(tgt3))))
        tgt = tgt + self.dropout4(tgt3)
        return tgt

    def forward(self, tgt, memory1, memory2,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos1: Optional[Tensor] = None,
                pos2: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory1, memory2, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, 
                                    pos1, pos2, query_pos)
        return self.forward_post(tgt, memory1, memory2, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, 
                                 pos1, pos2, query_pos)


class DecoderEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # exchanging for glove embeddings:
        self.word_embeddings = nn.Embedding.from_pretrained(config.pre_embed, padding_idx=config.pad_token_id)
        self.word_embeddings.requires_grad=True
        #self.word_embeddings = nn.Embedding(
        #    config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x.to(self.config.device))
        position_embeds = self.position_embeddings(position_ids.to(self.config.device))

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def build_transformer_double_encoder(config):
    return Transformer(
        config,
        d_model=config.hidden_dim,
        dropout=config.dropout,
        nhead=config.nheads,
        dim_feedforward=config.dim_feedforward,
        num_encoder_layers=config.enc_layers,
        num_decoder_layers=config.dec_layers,
        normalize_before=config.pre_norm,
        return_intermediate_dec=False,
        cnn_encoder_type=config.encoder_type
    )
