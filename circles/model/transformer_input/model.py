import torch
import torch.nn as nn
import math
from config import config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=config.MAX_SEQ_LENGTH):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config.DROPOUT)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class CircleTransformer(nn.Module):
    def __init__(self):
        super(CircleTransformer, self).__init__()

        self.coord_embedding = nn.Linear(2, config.D_MODEL)
        self.pos_encoder = PositionalEncoding(config.D_MODEL)

        self.transformer = nn.Transformer(
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT
        )

        self.output_layer = nn.Linear(config.D_MODEL, 2)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src_emb = self.coord_embedding(src)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.coord_embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)

        output = self.transformer(
            src_emb.transpose(0, 1),  # Transformer expects (seq_len, batch_size, feature_dim)
            tgt_emb.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        output = self.output_layer(output.transpose(0, 1))  # (batch_size, seq_len, 2)
        return output

    @torch.no_grad()
    def predict(self, partial_points, max_total_points=config.MAX_SEQ_LENGTH):
        self.eval()
        device = config.DEVICE

        partial_points = partial_points.to(device)
        batch_size = partial_points.size(0)
        input_len = partial_points.size(1)

        src_emb = self.coord_embedding(partial_points)
        src_emb = self.pos_encoder(src_emb)

        memory = self.transformer.encoder(src_emb.transpose(0, 1))

        generated = torch.zeros((batch_size, 1, 2), device=device)

        predictions = []
        for _ in range(max_total_points - input_len):
            tgt_emb = self.coord_embedding(generated)
            tgt_emb = self.pos_encoder(tgt_emb)

            output = self.transformer.decoder(tgt_emb.transpose(0, 1), memory)
            output = self.output_layer(output.transpose(0, 1))

            next_point = output[:, -1:, :]  # Last predicted point
            generated = torch.cat((generated, next_point), dim=1)
            predictions.append(next_point)

        predictions = torch.cat(predictions, dim=1)
        full_sequence = torch.cat([partial_points, predictions], dim=1)
        return full_sequence
