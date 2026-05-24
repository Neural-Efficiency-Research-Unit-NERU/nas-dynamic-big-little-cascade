"""Tests for analytical FLOPs (MACs) estimation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nas.search_space import BlockGene, PairGenotype
from src.training.utils import estimate_pair_flops, cascade_flops


def _make_pair(little_blocks: list[BlockGene], big_blocks: list[BlockGene]) -> PairGenotype:
    return PairGenotype(little_blocks=little_blocks, big_blocks=big_blocks)


def test_flops_known_standard_conv():
    """Verify FLOPs for a known genotype with standard conv matches hand calculation.

    Little path (2 blocks, ch=8, k=3, layers=1, standard, no residual):
      Block 0: in_ch=3, out_ch=8, stride=2 -> h=16,w=16
        Standard conv: 8 * 3 * 9 * 256 = 55,296
      Block 1: in_ch=8, out_ch=8, stride=2 -> h=8,w=8
        Standard conv: 8 * 8 * 9 * 64 = 36,864
      Classifier: 8 * 10 = 80
      Total: 55,296 + 36,864 + 80 = 92,240

    Big path (4 blocks, ch=8, k=3, layers=1, standard, no residual):
      Block 0: 8 * 3 * 9 * 256 = 55,296
      Block 1: 8 * 8 * 9 * 64  = 36,864
      Block 2: 8 * 8 * 9 * 16  =  9,216
      Block 3: 8 * 8 * 9 * 4   =  2,304
      Classifier: 80
      Total: 55,296 + 36,864 + 9,216 + 2,304 + 80 = 103,760
    """
    block = BlockGene(channels=8, layers=1, kernel_size=3, conv_type="standard", use_residual=False)
    g = _make_pair([block, block], [block, block, block, block])
    flops = estimate_pair_flops(g)

    assert flops["little_flops"] == 92_240
    assert flops["big_flops"] == 103_760
    assert flops["total_flops"] == 92_240 + 103_760


def test_flops_depthwise_less_than_standard():
    """Depthwise separable should use fewer FLOPs than standard conv at same config."""
    std_block = BlockGene(channels=32, layers=1, kernel_size=3, conv_type="standard", use_residual=False)
    dws_block = BlockGene(channels=32, layers=1, kernel_size=3, conv_type="depthwise_separable", use_residual=False)

    g_std = _make_pair([std_block, std_block], [std_block] * 4)
    g_dws = _make_pair([dws_block, dws_block], [dws_block] * 4)

    assert estimate_pair_flops(g_dws)["total_flops"] < estimate_pair_flops(g_std)["total_flops"]


def test_flops_more_channels_more_flops():
    """Larger channel count => more FLOPs."""
    small = BlockGene(channels=8, layers=1, kernel_size=3, conv_type="standard", use_residual=False)
    large = BlockGene(channels=48, layers=1, kernel_size=3, conv_type="standard", use_residual=False)

    g_small = _make_pair([small, small], [small] * 4)
    g_large = _make_pair([large, large], [large] * 4)

    assert estimate_pair_flops(g_large)["total_flops"] > estimate_pair_flops(g_small)["total_flops"]


def test_cascade_flops_all_exit():
    """If exit_ratio=1.0 (all samples handled by little), cascade_flops == little_flops."""
    little_f, big_f = 100_000, 500_000
    assert cascade_flops(little_f, big_f, exit_ratio=1.0) == little_f


def test_cascade_flops_no_exit():
    """If exit_ratio=0.0 (all sent to big), cascade_flops == little + big."""
    little_f, big_f = 100_000, 500_000
    assert cascade_flops(little_f, big_f, exit_ratio=0.0) == little_f + big_f
