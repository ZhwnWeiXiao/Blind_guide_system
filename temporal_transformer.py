import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    """Simple temporal transformer for frame sequences."""
    def __init__(self, in_channels: int = 3, d_model: int = 128, nhead: int = 8,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 num_layers: int = 3, max_len: int = 100):
        super().__init__()
        # Feature embedding: global average pool then linear projection
        self.embed = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, d_model)
        )
        # Positional encoding parameter
        self.pos_embedding = nn.Parameter(torch.zeros(max_len, d_model))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output head
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (T, C, H, W)
        Returns:
            Tensor of shape (T, d_model) representing temporal features.
        """
        # x -> (T, d_model)
        x = self.embed(x)
        # Add positional encoding
        pos = self.pos_embedding[:x.size(0)]
        x = x + pos
        # Transformer expects batch dimension
        x = self.encoder(x.unsqueeze(0)).squeeze(0)
        # Output head
        return self.head(x)
