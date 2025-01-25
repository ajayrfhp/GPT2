"""
Test if pytorch can access GPU
"""

import torch
import numpy as np
from models import MultiHeadAttention, LayerNorm, FeedForward, TransformerBlock
from gpt2 import GPT2
from utils import get_sum_parameters_of_model, get_memory_footprint_of_model

# pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


def test_gpu_presence():
    """Test presence of GPU"""
    assert torch.cuda.is_available()


def test_mha():
    """Test MultiHeadAttention layer"""
    config = {"emb_dim": 768, "heads": 12}
    mha = MultiHeadAttention(config)

    x = torch.randn(2, 10, 768)
    out = mha(x)

    expected_params = 4 * 768 * 768 / (1e6)

    assert out.shape == x.shape, f"Expected {x.shape} but got {out.shape}"

    assert np.allclose(
        get_sum_parameters_of_model(mha, millions=True), expected_params, atol=0.1
    ), f"Expected {expected_params} params"  # 4 * 768 * 768

    self_attention_matrix = mha.self_attention_matrix

    assert torch.allclose(
        self_attention_matrix.sum(dim=-1),
        torch.ones_like(self_attention_matrix.sum(dim=-1)),
    ), "Attention scores should sum to 1"

    assert torch.allclose(
        self_attention_matrix, torch.tril(self_attention_matrix, diagonal=1)
    ), "Attention scores should be causal"


def test_layer_norm():
    """Test layer norm"""
    config = {"emb_dim": 768, "heads": 12}
    ln = LayerNorm(config)

    x = torch.randn(2, 10, 768)
    out = ln(x)

    assert out.shape == x.shape, f"Expected {x.shape} but got {out.shape}"

    actual_mean, actual_std = out.mean(dim=-1), out.std(dim=-1)

    assert torch.allclose(
        actual_mean, torch.zeros_like(actual_mean), atol=0.1
    ), "Mean should be 0"

    assert torch.allclose(
        actual_std, torch.ones_like(actual_std), atol=0.1
    ), "Std should be 1"


def test_feed_forward():
    """Test feed forward layer"""
    config = {"emb_dim": 768, "heads": 12}
    ff = FeedForward(config)

    x = torch.randn(2, 10, 768)
    out = ff(x)

    assert out.shape == x.shape, f"Expected {x.shape} but got {out.shape}"

    actual_params = get_sum_parameters_of_model(ff, millions=True)
    expected_params = 8 * 768 * 768 / 1e6

    assert np.allclose(
        actual_params, expected_params, atol=0.1
    ), f"Expected {expected_params} params"  # 8 * 768 * 768


def test_transformer():
    """Tests transformer block"""
    config = {"emb_dim": 768, "heads": 12, "drop_out": 0.1}
    transformer = TransformerBlock(config)

    x = torch.randn(2, 10, 768)
    out = transformer(x)

    assert out.shape == x.shape, f"Expected {x.shape} but got {out.shape}"

    actual_params = get_sum_parameters_of_model(transformer, millions=True)
    expected_params = 7  # (4 * 768 * 768 + 8 * 768 * 768) / 1e6

    assert np.allclose(
        actual_params, expected_params, atol=0.1
    ), f"Expected {expected_params} params"


def test_GPT2():
    """Tests GPT2"""
    config = {
        "emb_dim": 768,
        "heads": 12,
        "drop_out": 0.1,
        "vocab_size": 50257,
        "layers": 12,
        "context_length": 1024,
        "device": "cpu",
    }
    gpt2 = GPT2(config)

    x = torch.randint(0, 50257, (2, 10))
    out = gpt2(x)

    assert out.shape == (2, 10, 50257), f"Expected (2, 10, 50257) but got {out.shape}"

    expected_params = 163  # 163 million
    actual_params = get_sum_parameters_of_model(gpt2, millions=True)

    assert np.allclose(
        actual_params, expected_params, atol=0.1
    ), f"Expected {expected_params} params"

    actual_memory = get_memory_footprint_of_model(gpt2)

    print(f"Memory footprint: {actual_memory} MB")


def test_gpt2_on_gpu():
    """Tests GPT2 on GPU"""
    device = torch.device("cuda")
    config = {
        "emb_dim": 768,
        "heads": 12,
        "drop_out": 0.1,
        "vocab_size": 50257,
        "layers": 12,
        "context_length": 1024,
        "device": device,
    }
    gpt2 = GPT2(config)
    x_on_gpu = torch.randint(0, 50257, (10, 10)).cuda()
    gpt2_on_gpu = gpt2.cuda()
    _ = gpt2_on_gpu(x_on_gpu)

    # Get memory usage from NVIDIA-SMI
    free_memory, total_memory = torch.cuda.mem_get_info()
    used_memory = total_memory - free_memory

    print(f"Used memory: {used_memory / 1e6} MB")
    print(f"Total memory: {total_memory / 1e6} MB")
