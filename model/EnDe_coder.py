"""
    This is a encoder-decoder structure model
"""

import torch.nn as nn


class EnDecoder(nn.Module):

    def __init__(self):
        super(EnDecoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

        def forward(self, x):
            """
                input: x[b, 1, 28, 28] ->
            """
            batch_size = x.size(0)
            x = x.view(batch_size, 784)
            x = self.encoder(x)
            x = self.decoder(x)
            x = x.view(batch_size, 1, 28, 28)
            return x
