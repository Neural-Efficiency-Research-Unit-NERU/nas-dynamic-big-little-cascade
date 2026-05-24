"""Two co-searched paired models for cascade dynamic inference."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nas.search_space import PairGenotype, INPUT_CHANNELS


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride=stride,
            padding=kernel_size // 2, groups=in_ch, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)), inplace=True)
        x = F.relu(self.bn2(self.pointwise(x)), inplace=True)
        return x


class ConvBlock(nn.Module):
    """Stack of conv layers (standard or depthwise separable) with optional residual."""

    def __init__(self, in_ch: int, out_ch: int, n_layers: int, kernel_size: int,
                 conv_type: str, use_residual: bool, stride: int = 1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            s = stride if i == 0 else 1
            c_in = in_ch if i == 0 else out_ch
            if conv_type == "depthwise_separable":
                layers.append(DepthwiseSeparableConv(c_in, out_ch, kernel_size, stride=s))
            else:
                layers.extend([
                    nn.Conv2d(c_in, out_ch, kernel_size, stride=s,
                              padding=kernel_size // 2, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ])
        self.layers = nn.Sequential(*layers)

        self.use_residual = use_residual and (in_ch == out_ch) and (stride == 1)
        self.shortcut = nn.Identity()
        if use_residual and not self.use_residual and (stride != 1 or in_ch != out_ch):
            self.use_residual = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.layers(x)
        if self.use_residual:
            out = out + self.shortcut(x)
        return out


class CascadePair(nn.Module):
    """Two fully independent co-searched models for cascade dynamic inference."""

    def __init__(self, genotype: PairGenotype, num_classes: int = 10):
        super().__init__()
        self.genotype = genotype

        # LittleModel: 2 searchable blocks + classifier
        # All blocks use stride=2
        little_blocks: list[nn.Module] = []
        in_ch = INPUT_CHANNELS
        for gene in genotype.little_blocks:
            little_blocks.append(ConvBlock(
                in_ch, gene.channels, gene.layers, gene.kernel_size,
                gene.conv_type, gene.use_residual, stride=2,
            ))
            in_ch = gene.channels
        little_blocks.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes),
        ])
        self.little = nn.Sequential(*little_blocks)

        # BigModel: 4 searchable blocks + classifier
        big_blocks: list[nn.Module] = []
        in_ch = INPUT_CHANNELS
        for gene in genotype.big_blocks:
            big_blocks.append(ConvBlock(
                in_ch, gene.channels, gene.layers, gene.kernel_size,
                gene.conv_type, gene.use_residual, stride=2,
            ))
            in_ch = gene.channels
        big_blocks.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes),
        ])
        self.big = nn.Sequential(*big_blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        little_logits = self.little(x)
        big_logits = self.big(x)
        return little_logits, big_logits

    @torch.no_grad()
    def cascade_inference(
        self, x: torch.Tensor, threshold: float = 0.8, temperature: float = 1.0,
    ) -> tuple[torch.Tensor, float]:
        """Run cascade: use little prediction if confident, else big."""
        little_out, big_out = self.forward(x)
        confidence = torch.softmax(little_out / temperature, dim=1).max(dim=1).values
        little_pred = little_out.argmax(dim=1)
        big_pred = big_out.argmax(dim=1)
        use_little = confidence > threshold
        predictions = torch.where(use_little, little_pred, big_pred)
        exit_ratio = use_little.float().mean().item()
        return predictions, exit_ratio
