"""Unit tests for model forward passes (skipped if torch is unavailable)."""

import pytest

torch = pytest.importorskip("torch")

from src.models import (  # noqa: E402
    TextCNN,
    BiGRUClassifier,
    SingleHeadAttentionClassifier,
    MultiHeadAttentionClassifier,
    TransformerEncoderClassifier,
    build_torch_model,
)

VOCAB, SEQ, BATCH = 50, 32, 4


def _dummy_batch():
    x = torch.randint(1, VOCAB, (BATCH, SEQ))
    x[:, SEQ // 2:] = 0  # simulate padding
    return x


@pytest.mark.parametrize("model", [
    TextCNN(VOCAB),
    BiGRUClassifier(VOCAB),
    SingleHeadAttentionClassifier(VOCAB),
    MultiHeadAttentionClassifier(VOCAB, num_heads=4),
    TransformerEncoderClassifier(VOCAB, num_layers=2, num_heads=4, pos_encoding="sinusoidal"),
    TransformerEncoderClassifier(VOCAB, num_layers=1, num_heads=4, pos_encoding="learnable"),
    TransformerEncoderClassifier(VOCAB, num_layers=2, num_heads=4, pos_encoding="none"),
])
def test_forward_shape_and_no_nan(model):
    out = model(_dummy_batch())
    assert out.shape == (BATCH, 2)
    assert not torch.isnan(out).any()


def test_fully_padded_input_is_nan_safe():
    model = MultiHeadAttentionClassifier(VOCAB)
    x = torch.zeros(2, SEQ, dtype=torch.long)
    assert not torch.isnan(model(x)).any()


def test_build_torch_model_unknown_arch_raises():
    with pytest.raises(ValueError):
        build_torch_model("not_a_model", VOCAB)
