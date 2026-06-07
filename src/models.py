"""Model definitions: vocabulary, dataset, and the architecture ladder.

Rungs 2-3 are a TextCNN and a BiGRU. Rungs 4-6 are the from-scratch
attention ladder (single-head, multi-head, and an N-layer Transformer
encoder), all implemented directly from query/key/value projections so
the attention mechanism is built by hand rather than imported.

This module requires PyTorch. It is imported only by the training
pipeline, not by the lightweight metric/data tests.
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from . import config


class SimpleVocab:
    """Minimal whitespace tokenizer and word-to-id vocabulary.

    Index 0 is reserved for ``<PAD>`` and index 1 for ``<UNK>``.
    """

    def __init__(self, min_freq=1):
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.itos = ["<PAD>", "<UNK>"]
        self.min_freq = min_freq

    def tokenize(self, text):
        """Lowercase-split a string into tokens."""
        return str(text).lower().split()

    def fit(self, texts):
        """Build the vocabulary from an iterable of training texts."""
        counts = {}
        for text in texts:
            for tok in self.tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
        for tok, cnt in counts.items():
            if cnt >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, text, max_len):
        """Encode text to a fixed-length list of token ids (pad/truncate)."""
        ids = [self.stoi.get(tok, 1) for tok in self.tokenize(text)]
        ids = ids[:max_len]
        ids += [0] * max(0, max_len - len(ids))
        return ids


class TorchTextDataset(Dataset):
    """Dataset yielding ``(token_ids, label)`` tensor pairs."""

    def __init__(self, texts, labels, vocab, max_len):
        self.X = [vocab.encode(t, max_len=max_len) for t in texts]
        self.y = list(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(int(self.y[idx]), dtype=torch.long),
        )


class TextCNN(nn.Module):
    """Convolutional text classifier (rung 2): multi-kernel + max-pool."""

    def __init__(self, vocab_size, embed_dim=128, num_filters=64,
                 kernel_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 2)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)
        outs = [torch.max(torch.relu(conv(emb)), dim=2).values for conv in self.convs]
        out = self.dropout(torch.cat(outs, dim=1))
        return self.fc(out)


class BiGRUClassifier(nn.Module):
    """Bidirectional GRU text classifier (rung 3) with max-pool over time."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        pooled = torch.max(out, dim=1).values
        return self.fc(self.dropout(pooled))


def masked_mean(x, mask):
    """Mean-pool token vectors over valid (non-pad) positions.

    Parameters
    ----------
    x : torch.Tensor
        ``(batch, seq, dim)`` token representations.
    mask : torch.Tensor
        ``(batch, seq)`` boolean, True for real tokens.

    Returns
    -------
    torch.Tensor
        ``(batch, dim)`` mean over valid positions.
    """
    m = mask.unsqueeze(-1).float()
    summed = (x * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1.0)
    return summed / counts


class SingleHeadAttentionClassifier(nn.Module):
    """Rung 4: one self-attention head implemented from QKV by hand."""

    def __init__(self, vocab_size, embed_dim=128, dropout=0.3, **_):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        mask = x != 0
        emb = self.embedding(x)
        scores = (self.q(emb) @ self.k(emb).transpose(1, 2)) / self.scale
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        ctx = torch.softmax(scores, dim=-1) @ self.v(emb)
        return self.fc(self.dropout(masked_mean(ctx, mask)))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with a key padding mask."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must divide num_heads"
        self.h = num_heads
        self.dk = embed_dim // num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.o = nn.Linear(embed_dim, embed_dim)
        self.scale = self.dk ** 0.5

    def forward(self, x, mask):
        B, T, E = x.shape

        def split(t):
            return t.view(B, T, self.h, self.dk).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = (q @ k.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(~mask[:, None, None, :], -1e9)
        ctx = torch.softmax(scores, dim=-1) @ v
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, E)
        return self.o(ctx)


class MultiHeadAttentionClassifier(nn.Module):
    """Rung 5: a single multi-head attention block plus a pooling head."""

    def __init__(self, vocab_size, embed_dim=128, num_heads=4, dropout=0.3, **_):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        mask = x != 0
        ctx = self.attn(self.embedding(x), mask)
        return self.fc(self.dropout(masked_mean(ctx, mask)))


class EncoderBlock(nn.Module):
    """One Transformer encoder block: MHA + FFN, each with residual + norm."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


def sinusoidal_positions(max_len, dim):
    """Return a ``(max_len, dim)`` sinusoidal positional-encoding matrix."""
    pe = torch.zeros(max_len, dim)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerEncoderClassifier(nn.Module):
    """Rung 6: N stacked encoder blocks with a positional-encoding toggle.

    Parameters
    ----------
    pos_encoding : str
        ``'sinusoidal'``, ``'learnable'``, or ``'none'``.
    num_layers : int
        Number of stacked encoder blocks (e.g. 1, 2, 4).
    """

    def __init__(self, vocab_size, embed_dim=128, num_heads=4, ff_dim=256,
                 num_layers=2, dropout=0.3, pos_encoding="sinusoidal",
                 max_len=None, **_):
        super().__init__()
        max_len = config.MAX_LENGTH if max_len is None else max_len
        self.pos_encoding = pos_encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pos_encoding == "sinusoidal":
            self.register_buffer("pe", sinusoidal_positions(max_len, embed_dim))
        elif pos_encoding == "learnable":
            self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.blocks = nn.ModuleList(
            [EncoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        mask = x != 0
        seq_len = x.size(1)
        emb = self.embedding(x)
        if self.pos_encoding == "sinusoidal":
            emb = emb + self.pe[:seq_len].unsqueeze(0).to(x.device)
        elif self.pos_encoding == "learnable":
            pos = torch.arange(seq_len, device=x.device)
            emb = emb + self.pos_emb(pos).unsqueeze(0)
        h = self.dropout(emb)
        for block in self.blocks:
            h = block(h, mask)
        return self.fc(self.dropout(masked_mean(h, mask)))


def build_torch_model(arch, vocab_size, **kwargs):
    """Instantiate a torch model by architecture key.

    Parameters
    ----------
    arch : str
        One of ``text_cnn``, ``bigru``, ``single_head_attn``,
        ``multi_head_attn``, ``transformer_encoder``.
    vocab_size : int
        Size of the vocabulary (embedding rows).
    **kwargs
        Architecture-specific hyperparameters (``num_heads``,
        ``num_layers``, ``pos_encoding``, ``dropout``, ...).

    Returns
    -------
    torch.nn.Module

    Raises
    ------
    ValueError
        If ``arch`` is not recognised.
    """
    if arch == "text_cnn":
        return TextCNN(vocab_size, **kwargs)
    if arch == "bigru":
        return BiGRUClassifier(vocab_size, **kwargs)
    if arch == "single_head_attn":
        return SingleHeadAttentionClassifier(vocab_size, **kwargs)
    if arch == "multi_head_attn":
        return MultiHeadAttentionClassifier(vocab_size, **kwargs)
    if arch == "transformer_encoder":
        return TransformerEncoderClassifier(vocab_size, **kwargs)
    raise ValueError(f"Unknown arch: {arch}")
