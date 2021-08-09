import torch.nn as nn

class  Prediction(nn.Sequential):
    """
        >>> tensor = torch.randn(8, 3, 12, 12)
        >>> tensor = Prediction(3)(tensor)
        >>> tensor.size()
        >>> (8, 1, 12, 12)
    """
    def __init__(self, inc, ratio = 4):
        super().__init__(            
            nn.Conv2d(inc, inc, 3, 1, 1),
            # nn.Conv2d(inc, inc // ratio, 3, 1, 1),
            # nn.Conv2d(inc // ratio, 1, 1),
            nn.Conv2d(inc, 1, 1),
            nn.Sigmoid()
        )
        self.inc = inc