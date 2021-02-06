"""
    For a three-level directory structure, we can use ImageFolder module
    which in torchvision.datasets.folder

    root
      ├── train
      │     ├── class1
      │     │     ├── 00001.jpg
      │     │     ├── 00002.jpg
      │     │     └── 00003.jpg
      │     │
      │     └── class2
      │             ├── 00001.jpg
      │             ├── 00002.jpg
      │             └── 00003.jpg
      └── test
            ├── class1
            │     ├── 00001.jpg
            │     ├── 00002.jpg
            │     └── 00003.jpg
            │
            └── class2
                    ├── 00001.jpg
                    ├── 00002.jpg
                    └── 00003.jpg

"""

from torchvision.datasets.folder import ImageFolder

"""
    ImageFolder:
        def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
        ):
    parameters:
        root: root + train/test
        transform: transform for your sample
        target_transform: transform for your label
        loader: unknown
        is_valid_file: unknown
    
    If you often design model for classification, I advice you write 
    your own data loader template, the detail of ImageFolder is not easy to know and 
    whether the implement of ImageFolder will be changed with the iteration 
    of pytorch is uncertain either
"""