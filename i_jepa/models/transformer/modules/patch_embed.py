import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        input_channels: int = 3,
        output_dims: int = 256,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.output_dims = output_dims

        self.num_patches = (image_size // patch_size) ** 2

        self.layer_norm = nn.LayerNorm(output_dims)

        self.projection = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_dims,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.projection(img)
        x = x.view(x.size(0), self.output_dims, -1).transpose(1, 2)
        x = self.layer_norm(x)

        return x
