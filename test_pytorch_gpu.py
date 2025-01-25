"""
Test if pytorch can access GPU
"""
import torch

# pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


def test_gpu_presence():
    assert torch.cuda.is_available()
