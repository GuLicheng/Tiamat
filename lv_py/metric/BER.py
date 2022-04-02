# https://github.com/xw-hu/CUHK-Shadow

from lv_utility.image import ImageHandle
import numpy as np
import cv2 as cv
import os

def invoke(callable, *args):
    return [_ for _ in map(callable, args)]

def concat_path(root):
    return [_ for _ in map(lambda x: root + '/' + x, os.listdir(root))]

def calculateBER(dir1, dir2, msg="", threshold = 127):

    dir1, dir2 = invoke(concat_path, dir1, dir2)

    dir1.sort()
    dir2.sort()

    TP, TN, POS, NEG = 0, 0, 0, 0
    threshold = 127  # change threshold from 125 to 127, [0, 127] is 0 and [128, 255] is 1

    FP, FN = 0, 0

    for img1, img2 in zip(dir1, dir2):

        def imread(arg):
            img = cv.imread(arg, cv.IMREAD_GRAYSCALE).flatten()
            return img
            
        # arr1, arr2 = invoke(imread, img1, img2)
        import copy
        arr1 = cv.imread(img1, cv.IMREAD_GRAYSCALE)
        arr2 = cv.imread(img2, cv.IMREAD_GRAYSCALE)
        arr2 = cv.resize(arr2, (arr1.shape[1], arr1.shape[0]))
        arr1 = arr1.flatten()
        arr2 = arr2.flatten()
        
        posPoints = arr1 > threshold
        negPoints = arr1 <= threshold

        countPos = posPoints.sum()
        countNeg = negPoints.sum()

        tp = posPoints & (arr2 > threshold)
        countTP = tp.sum()

        tn = negPoints & (arr2 <= threshold)
        countTN = tn.sum()


        FP += ((posPoints) & (arr2 > threshold)).sum()
        FN += ((negPoints) & (arr2 < threshold)).sum()

        # type is np.int32. Convert to python build-in type `int` if overflow
        # TP += countTP.tolist()
        TP += countTP
        TN += countTN
        POS += countPos
        NEG += countNeg

    # print(invoke(type, TP, TN, POS, NEG))

    posAcc = TP / POS
    negAcc = TN / NEG

    BER = 0.5 * (2 - posAcc - negAcc)

    acc_final = (TP + TN) / (POS + NEG)
    final_BER = BER * 100
    pErr = (1 - posAcc) * 100
    nErr = (1 - negAcc) * 100

    print(f"info: {msg} BER: {final_BER:.2f}, pErr: {pErr:.2f}, nErr: {nErr:.2f}, acc: {acc_final:.4f}, ")

    return acc_final, final_BER, pErr, nErr

import numpy as np
import os

import pydensecrf.densecrf as dcrf

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')


def main():
    GTS = "G:/DataSet/SBUshadow/SBUTestRename/ShadowMasks"
    MASKS1 = r"D:\MY\SOD\Swin-Transformer\PredMask\32500"
    MASKS2 = r"D:\MY\SOD\Swin-Transformer\fix32500"
    MASKS4 = r"D:\MY\SOD\Tiamat\fix35000"
    MASKS3 = r"D:\MY\SOD\MTMT\SaliencyMap\results_of_SBU_ISTD_UCF\SBU_crf"
    MASKS5 = r"G:\Masks\crf"
    calculateBER(GTS, MASKS5)
    # calculateBER(GTS, MASKS2)
    # calculateBER(GTS, MASKS3)
    # for i in range(0, 28000 + 1, 2000):
        # calculateBER(dir1=GTS, dir2=rf"G:\Masks\{i}", msg=f"{i}")


if __name__ == "__main__":
    main()
    # gts = ImageHandle.files(r"G:\Masks\24000")
    # images = ImageHandle.files(r"G:\DataSet\SBUshadow\SBUTestRename\ShadowImages")
    # for img, anno in zip(images, gts):
    #     name = anno.split('/')[-1]
    #     img = cv.imread(img, cv.IMREAD_COLOR)
    #     anno = cv.imread(anno, cv.IMREAD_GRAYSCALE)
    #     print(img.shape, anno.shape)
    #     res = crf_refine(img, anno)
    #     cv.imwrite(f"G:/Masks/crf/{name}", res)