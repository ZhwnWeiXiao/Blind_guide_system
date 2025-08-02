import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """Temporal transformer for sequences of video frames.

    Each frame is split into non-overlapping patches which are embedded and
    processed by a stack of Transformer encoder layers. The module outputs a
    temporal feature vector for every frame in the sequence. This implementation
    is lightweight and suited for single-camera inputs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_layers: int = 3,
        patch_size: int = 16,
        max_len: int = 100,
    ) -> None:
        super().__init__()

        # Patch embedding: conv projection + flatten
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)  # (N, d_model, num_patches)

        # Positional encoding for the temporal dimension
        self.pos_embedding = nn.Parameter(torch.zeros(max_len, d_model))

        # Transformer encoder operating on frame-level embeddings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head mapping to temporal features
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute temporal features from a sequence of frames.

        Args:
            x: Tensor of shape ``(T, C, H, W)`` containing ``T`` RGB frames.

        Returns:
            Tensor of shape ``(T, d_model)`` with the temporal embedding for
            each frame in the input sequence.
        """

        # Convert frames to patch embeddings -> (T, d_model, num_patches)
        x = self.proj(x)
        x = self.flatten(x).transpose(1, 2)  # (T, num_patches, d_model)

        # Aggregate patch tokens into a single feature per frame
        x = x.mean(dim=1)

        # Add learnable temporal positional encoding
        pos = self.pos_embedding[: x.size(0)]
        x = x + pos

        # Transformer expects a batch dimension (sequence length, features)
        x = self.encoder(x.unsqueeze(0)).squeeze(0)

        # Final projection
        return self.head(x)
