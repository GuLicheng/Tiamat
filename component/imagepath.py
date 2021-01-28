"""
    2020/11/30 - now
    This module provide image file path for your data loader
"""
from typing import *

"""the Second Directory exported finally"""
# training set path
SECONDARY_DIRECTORY_TRAIN_ROOT: str = f"D:\MY\DataSet\DUTS\DUTS-TR"
SECONDARY_DIRECTORY_TRAIN_PATHS: Tuple[str, ...] = (
    f"/DUTS-TR-Image",
    f"/DUTS-TR-Mask",
    f"/DUTS-TR-Edge"
)
SECONDARY_DIRECTORY_TRAIN_SUFFIX: Tuple[str, ...] = (
    ".jpg",
    ".png",
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

"""ths Three-Level Directory exported finally"""
