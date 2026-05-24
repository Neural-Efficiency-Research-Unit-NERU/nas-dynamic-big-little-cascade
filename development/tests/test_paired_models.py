"""Tests for two co-searched paired models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.paired_models import CascadePair, DepthwiseSeparableConv
from src.nas.search_space import random_pair_genotype
from src.training.utils import count_parameters


def test_depthwise_separable_conv_shape():
    layer = DepthwiseSeparableConv(16, 32, kernel_size=3, stride=1)
    x = torch.randn(2, 16, 32, 32)
    out = layer(x)
    assert out.shape == (2, 32, 32, 32)


def test_depthwise_separable_conv_stride2():
    layer = DepthwiseSeparableConv(16, 32, kernel_size=3, stride=2)
    x = torch.randn(2, 16, 32, 32)
    out = layer(x)
    assert out.shape == (2, 32, 16, 16)


def test_cascade_pair_forward_shape():
    genotype = random_pair_genotype()
    model = CascadePair(genotype)
    x = torch.randn(4, 3, 32, 32)
    little_out, big_out = model(x)
    assert little_out.shape == (4, 10)
    assert big_out.shape == (4, 10)


def test_cascade_pair_param_count_reasonable():
    genotype = random_pair_genotype()
    model = CascadePair(genotype)
    n_params = count_parameters(model)
    assert n_params < 500_000, f"Model too large: {n_params:,} params"
    assert n_params > 100, f"Model suspiciously small: {n_params} params"


def test_cascade_inference():
    genotype = random_pair_genotype()
    model = CascadePair(genotype)
    model.eval()
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        preds, exit_ratio = model.cascade_inference(x, threshold=0.5)
    assert preds.shape == (8,)
    assert 0.0 <= exit_ratio <= 1.0


def test_multiple_genotypes_produce_different_sizes():
    sizes = set()
    for _ in range(10):
        g = random_pair_genotype()
        m = CascadePair(g)
        sizes.add(count_parameters(m))
    assert len(sizes) > 1


def test_little_and_big_are_independent():
    genotype = random_pair_genotype()
    model = CascadePair(genotype)
    little_params = set(id(p) for p in model.little.parameters())
    big_params = set(id(p) for p in model.big.parameters())
    assert len(little_params & big_params) == 0
