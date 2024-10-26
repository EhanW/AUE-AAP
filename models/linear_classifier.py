from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, n_classes: int = 10):
        super().__init__()
        self.enc = encoder
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.lin = nn.Linear(self.feature_dim, self.n_classes)

    def forward(self, x):
        return self.lin(self.enc(x))
