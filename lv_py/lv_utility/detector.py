import os
import cv2 as cv
import numpy as np

IMAGE_ROOT = r"G:\DataSet\SBUshadow\SBUTestRename\ShadowImages"
MASK_ROOT = r"G:\DataSet\SBUshadow\SBUTestRename\ShadowMasks"
# PRED_ROOT = r"D:\MY\SOD\TransUNet\pred"
ORIGIN_MASKS = r"D:\MY\SOD\TransUNet\origin_mask"
ORIGIN_HANDLE = r"D:\MY\SOD\TransUNet\pred_afterhandle"
PRED_ADDEDGE_ROOT = r"D:\MY\SOD\TransUNet\pred_addedge"
TRANSUNET = r"D:\MY\SOD\TransUNet\pred_transUnet"
# TRANSUNET_HANDLE = r"D:\MY\SOD\TransUNet\transUnet_handle"
RGB_SALIENCY = r"D:\MY\SOD\TransUNet\pred_rgb"
INVERSION = r"D:\MY\SOD\Tiamat\inversion_image"
RGB_INVERSION = r"D:\MY\SOD\TransUNet\pred_rgb_inversion"

SWIM_TRANSFORMER = r"D:\MY\SOD\Swin-Transformer\fix32500"
SWIM_TRANSFORMER_ADD_CHANNEL_ATTENTION = r"D:\MY\SOD\Swin-Transformer\20210628prediction\37500"
PVT = r"D:\MY\SOD\PVT\Masks\28000"
paths = [IMAGE_ROOT, MASK_ROOT, PVT, ORIGIN_HANDLE, TRANSUNET, SWIM_TRANSFORMER]

def create_file_list(root: str):
    files = os.listdir(root)
    return [os.path.join(root, file) for file in files]

images = [create_file_list(root) for root in paths]
# start = 100

def zip_partial(start, *args):
    return zip(*[arg[start:] for arg in args])

def read_mask(path):
    mask = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

cur = 0

for args in zip_partial(cur, *images):
    print(args)
    image = cv.imread(args[0], cv.IMREAD_COLOR)

    # mask = cv.imread(args[1], cv.IMREAD_GRAYSCALE)
    # mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    # pred = cv.imread(args[2], cv.IMREAD_GRAYSCALE)
    # pred = cv.cvtColor(pred, cv.COLOR_GRAY2BGR)

    # origin_masks = cv.imread(args[3], cv.IMREAD_GRAYSCALE)
    # origin_masks = cv.cvtColor(origin_masks, cv.COLOR_GRAY2BGR)

    mask = read_mask(args[1])

    pred = read_mask(args[2])

    origin_mask = read_mask(args[3])

    transUnet = read_mask(args[4])

    transUnet_after_handle = read_mask(args[5])



    row1 = np.hstack([image, mask])
    row2 = np.hstack([pred, origin_mask])
    row3 = np.hstack([transUnet, transUnet_after_handle])
    images = np.vstack([row1, row2, row3])
    windowname = f"Image {cur}"
    cur += 1
    cv.namedWindow(windowname, cv.WINDOW_NORMAL)
    cv.resizeWindow(windowname, 1200, 900)
    cv.imshow(windowname, images)
    cv.waitKey(0)
    
