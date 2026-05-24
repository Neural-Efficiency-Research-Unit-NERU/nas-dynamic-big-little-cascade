"""NAS search space for co-searched little/big model pair (Depth Explorer)."""
import random
from dataclasses import dataclass, field


@dataclass
class BlockGene:
    channels: int
    layers: int
    kernel_size: int
    conv_type: str  # "standard" or "depthwise_separable"
    use_residual: bool


# Asymmetric channel options
LITTLE_CHANNELS = [8, 12, 16, 24, 32]
BIG_CHANNELS = [16, 24, 32, 48, 64]

# Shared options
LAYERS_OPTIONS = [1, 2]
KERNEL_OPTIONS = [3, 5]
CONV_TYPE_OPTIONS = ["standard", "depthwise_separable"]

# Architecture depth ranges (searchable)
LITTLE_BLOCKS_MIN = 1
LITTLE_BLOCKS_MAX = 3
BIG_BLOCKS_MIN = 2
BIG_BLOCKS_MAX = 5

# Threshold range retained for vector compatibility and post-hoc sweeps.
# Joint CIFAR search can override it with search.fixed_threshold.
THRESHOLD_MIN = 0.60
THRESHOLD_MAX = 0.95

# Architecture constants
INPUT_CHANNELS = 3


@dataclass
class PairGenotype:
    little_blocks: list[BlockGene]
    big_blocks: list[BlockGene]
    threshold: float = 0.80

    def to_dict(self) -> dict:
        def _block_to_dict(b: BlockGene) -> dict:
            return {
                "channels": b.channels,
                "layers": b.layers,
                "kernel_size": b.kernel_size,
                "conv_type": b.conv_type,
                "use_residual": b.use_residual,
            }
        return {
            "little": [_block_to_dict(b) for b in self.little_blocks],
            "big": [_block_to_dict(b) for b in self.big_blocks],
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PairGenotype":
        def _dict_to_block(bd: dict) -> BlockGene:
            return BlockGene(
                channels=bd["channels"],
                layers=bd["layers"],
                kernel_size=bd["kernel_size"],
                conv_type=bd["conv_type"],
                use_residual=bd["use_residual"],
            )
        return cls(
            little_blocks=[_dict_to_block(b) for b in d["little"]],
            big_blocks=[_dict_to_block(b) for b in d["big"]],
            threshold=d.get("threshold", 0.80),
        )


def _random_block(channel_options: list[int]) -> BlockGene:
    return BlockGene(
        channels=random.choice(channel_options),
        layers=random.choice(LAYERS_OPTIONS),
        kernel_size=random.choice(KERNEL_OPTIONS),
        conv_type=random.choice(CONV_TYPE_OPTIONS),
        use_residual=random.choice([True, False]),
    )


def random_pair_genotype() -> PairGenotype:
    n_little = random.randint(LITTLE_BLOCKS_MIN, LITTLE_BLOCKS_MAX)
    n_big = random.randint(BIG_BLOCKS_MIN, BIG_BLOCKS_MAX)
    threshold = random.uniform(THRESHOLD_MIN, THRESHOLD_MAX)
    return PairGenotype(
        little_blocks=[_random_block(LITTLE_CHANNELS) for _ in range(n_little)],
        big_blocks=[_random_block(BIG_CHANNELS) for _ in range(n_big)],
        threshold=threshold,
    )


def _encode_block(b: BlockGene, channel_options: list[int]) -> list[float]:
    return [
        float(channel_options.index(b.channels)),
        float(LAYERS_OPTIONS.index(b.layers)),
        float(KERNEL_OPTIONS.index(b.kernel_size)),
        float(CONV_TYPE_OPTIONS.index(b.conv_type)),
        1.0 if b.use_residual else 0.0,
    ]


def _decode_blocks(vec: list[float], start: int, n_blocks: int,
                   channel_options: list[int]) -> list[BlockGene]:
    blocks = []
    for i in range(n_blocks):
        offset = start + i * _GENES_PER_BLOCK
        ch_idx = int(round(vec[offset])) % len(channel_options)
        ly_idx = int(round(vec[offset + 1])) % len(LAYERS_OPTIONS)
        ks_idx = int(round(vec[offset + 2])) % len(KERNEL_OPTIONS)
        ct_idx = int(round(vec[offset + 3])) % len(CONV_TYPE_OPTIONS)
        res = vec[offset + 4] >= 0.5
        blocks.append(BlockGene(
            channels=channel_options[ch_idx],
            layers=LAYERS_OPTIONS[ly_idx],
            kernel_size=KERNEL_OPTIONS[ks_idx],
            conv_type=CONV_TYPE_OPTIONS[ct_idx],
            use_residual=res,
        ))
    return blocks


def pair_genotype_to_vector(g: PairGenotype) -> list[float]:
    """Encode PairGenotype as 43-dim vector for pymoo.

    Layout:
      dims 0-14:  little block slots 0,1,2 (3 max x 5 genes)
      dims 15-39: big block slots 0,1,2,3,4 (5 max x 5 genes)
      dim 40:     n_little_active
      dim 41:     n_big_active
      dim 42:     threshold

    Inactive block slots are padded by repeating the last active block.
    """
    vec: list[float] = []

    # Encode little blocks (pad to LITTLE_BLOCKS_MAX)
    for i in range(LITTLE_BLOCKS_MAX):
        if i < len(g.little_blocks):
            vec.extend(_encode_block(g.little_blocks[i], LITTLE_CHANNELS))
        else:
            vec.extend(_encode_block(g.little_blocks[-1], LITTLE_CHANNELS))

    # Encode big blocks (pad to BIG_BLOCKS_MAX)
    for i in range(BIG_BLOCKS_MAX):
        if i < len(g.big_blocks):
            vec.extend(_encode_block(g.big_blocks[i], BIG_CHANNELS))
        else:
            vec.extend(_encode_block(g.big_blocks[-1], BIG_CHANNELS))

    # Meta-genes
    vec.append(float(len(g.little_blocks)))
    vec.append(float(len(g.big_blocks)))
    vec.append(g.threshold)

    return vec


def vector_to_pair_genotype(
    vec: list[float],
    threshold_override: float | None = None,
) -> PairGenotype:
    """Decode 43-dim vector back to PairGenotype.

    `threshold_override` keeps backwards-compatible vectors/CSVs while letting
    experiments fix the cascade threshold during architecture search.
    """
    # Meta-genes at the end
    n_little_raw = vec[_META_START]
    n_big_raw = vec[_META_START + 1]
    threshold_raw = vec[_META_START + 2]

    # Clamp and round active counts
    n_little = int(round(max(LITTLE_BLOCKS_MIN, min(LITTLE_BLOCKS_MAX, n_little_raw))))
    n_big = int(round(max(BIG_BLOCKS_MIN, min(BIG_BLOCKS_MAX, n_big_raw))))

    # Clamp threshold, unless the caller fixes it for architecture-only search.
    raw_threshold = threshold_raw if threshold_override is None else threshold_override
    threshold = max(THRESHOLD_MIN, min(THRESHOLD_MAX, raw_threshold))

    # Decode only the active blocks
    little_blocks = _decode_blocks(vec, start=0, n_blocks=n_little,
                                   channel_options=LITTLE_CHANNELS)
    big_blocks = _decode_blocks(vec, start=LITTLE_BLOCKS_MAX * _GENES_PER_BLOCK,
                                n_blocks=n_big, channel_options=BIG_CHANNELS)

    return PairGenotype(
        little_blocks=little_blocks,
        big_blocks=big_blocks,
        threshold=threshold,
    )


# Encoding constants
_GENES_PER_BLOCK = 5
_META_START = (LITTLE_BLOCKS_MAX + BIG_BLOCKS_MAX) * _GENES_PER_BLOCK  # 40

N_VAR = _META_START + 3  # 43

# Per-block bounds
_little_block_xl = [0, 0, 0, 0, 0]
_little_block_xu = [
    len(LITTLE_CHANNELS) - 1,
    len(LAYERS_OPTIONS) - 1,
    len(KERNEL_OPTIONS) - 1,
    len(CONV_TYPE_OPTIONS) - 1,
    1,
]
_big_block_xl = [0, 0, 0, 0, 0]
_big_block_xu = [
    len(BIG_CHANNELS) - 1,
    len(LAYERS_OPTIONS) - 1,
    len(KERNEL_OPTIONS) - 1,
    len(CONV_TYPE_OPTIONS) - 1,
    1,
]

XL = (
    _little_block_xl * LITTLE_BLOCKS_MAX
    + _big_block_xl * BIG_BLOCKS_MAX
    + [float(LITTLE_BLOCKS_MIN), float(BIG_BLOCKS_MIN), THRESHOLD_MIN]
)
XU = (
    _little_block_xu * LITTLE_BLOCKS_MAX
    + _big_block_xu * BIG_BLOCKS_MAX
    + [float(LITTLE_BLOCKS_MAX), float(BIG_BLOCKS_MAX), THRESHOLD_MAX]
)
