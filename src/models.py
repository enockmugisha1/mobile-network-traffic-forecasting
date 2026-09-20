from __future__ import annotations
import math
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, seq_len: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class _Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp
    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous()


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            _Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            _Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.skip(x))


class TCNForecaster(nn.Module):
    def __init__(self, seq_len: int, num_channels: int = 32,
                 num_levels: int = 4, kernel_size: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        channels = [1] + [num_channels] * num_levels
        self.network = nn.Sequential(*[
            _ResidualBlock(channels[i], channels[i+1],
                           kernel_size, 2**i, dropout)
            for i in range(num_levels)
        ])
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        y = self.network(x.permute(0, 2, 1))
        return self.fc(y[:, :, -1])


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerForecaster(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, dropout=dropout)
        self.encoder    = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                       dropout, batch_first=True),
            num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        return self.fc(self.encoder(x)[:, -1, :])
