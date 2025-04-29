import torch
import torch.nn as nn
import math
from config import config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=config.MAX_SEQ_LENGTH):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config.DROPOUT)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # Fixed parentheses
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

        # Input embedding layer (for 2D coordinates)
        self.coord_embedding = nn.Linear(2, config.D_MODEL)
        self.pos_encoder = PositionalEncoding(config.D_MODEL)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT
        )

        # Output layer
        self.output_layer = nn.Linear(config.D_MODEL, 2)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        # Embed coordinates
        src_embedded = self.coord_embedding(src)
        tgt_embedded = self.coord_embedding(tgt)

        # Add positional encoding
        src_embedded = self.pos_encoder(src_embedded)
        tgt_embedded = self.pos_encoder(tgt_embedded)

        # Transformer expects (seq_len, batch_size, d_model)
        src_embedded = src_embedded.permute(1, 0, 2)
        tgt_embedded = tgt_embedded.permute(1, 0, 2)

        # Create attention masks
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(config.DEVICE)

        # Transformer forward pass
        output = self.transformer(
            src_embedded, tgt_embedded,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask
        )

        # Output layer
        output = self.output_layer(output)

        # Return to (batch_size, seq_len, 2) shape
        output = output.permute(1, 0, 2)

        return output

    def predict(self, src, max_length=config.MAX_SEQ_LENGTH):
        self.eval()
        with torch.no_grad():
            # Embed source
            src_embedded = self.coord_embedding(src.unsqueeze(0))
            src_embedded = self.pos_encoder(src_embedded)
            src_embedded = src_embedded.permute(1, 0, 2)

            # Initialize target with start token
            tgt = torch.zeros(1, 1, 2).to(config.DEVICE)
            output_sequence = []

            for i in range(max_length):
                tgt_embedded = self.coord_embedding(tgt)
                tgt_embedded = self.pos_encoder(tgt_embedded)
                tgt_embedded = tgt_embedded.permute(1, 0, 2)

                tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(config.DEVICE)

                output = self.transformer(
                    src_embedded, tgt_embedded,
                    tgt_mask=tgt_mask
                )

                output = self.output_layer(output)
                next_point = output[-1, :, :]
                output_sequence.append(next_point)

                # Prepare next input
                tgt = torch.cat([tgt, next_point.unsqueeze(0)], dim=1)

            output_sequence = torch.cat(output_sequence, dim=0)
            return output_sequence
