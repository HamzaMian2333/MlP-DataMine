"""Simple multilayer perceptron for flat feature vectors."""

from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: tuple[int, ...] = (512, 256),
        num_classes: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, input_dim) -> logits: (batch, num_classes)
        return self.layers(x)
