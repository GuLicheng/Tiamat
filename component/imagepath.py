"""
    2020/11/30 - now
    This module provide image file path for your data loader
"""
from typing import *

"""the Second Directory exported finally"""
# training set path
SECONDARY_DIRECTORY_TRAIN_ROOT: str = f"D:\SOD\SOD\SOD"
SECONDARY_DIRECTORY_TRAIN_PATHS: Tuple[str, ...] = (
    f"\image",
    f"\gt"
)
SECONDARY_DIRECTORY_TRAIN_SUFFIX: Tuple[str, ...] = (
    ".jpg",
    ".png"
)

# testing set path
SECONDARY_DIRECTORY_TEST_ROOT: str = f"D:\MY\DataSet\DUTS\DUTS-TE"
SECONDARY_DIRECTORY_TEST_PATHS: Tuple[str, ...] = (
    f"/DUTS-TE-Image",
    f"/DUTS-TE-Mask",
)

SECONDARY_DIRECTORY_TEST_SUFFIX: Tuple[str, ...] = (
    ".jpg",
    ".png"
)

"""the Three-Level Directory exported finally"""
