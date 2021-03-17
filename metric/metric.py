import numpy as np

def cal_pr_mae_meanf(prediction, gt):
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    # make sure that image is same as ground truth
    """
    if prediction.shape != gt.shape:
        prediction = Image.fromarray(prediction).convert('L')
        gt_temp = Image.fromarray(gt).convert('L')
        prediction = prediction.resize(gt_temp.size)
        prediction = np.array(prediction)
    """

    # get 0-1 truth map from prediction and ground truth
    if prediction.max() == prediction.min():
        prediction = prediction / 255
    else:
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    hard_gt = np.zeros_like(gt)
    hard_gt[gt > 128] = 1

    # MAE
    mae = np.mean(np.abs(prediction - hard_gt))

    # MeanF
    threshold_fm = 2 * prediction.mean()
    if threshold_fm > 1:
        threshold_fm = 1
    binary = np.zeros_like(prediction)
    binary[prediction >= threshold_fm] = 1
    tp = (binary * hard_gt).sum()
    if tp == 0:
        meanf = 0
    else:
        pre = tp / binary.sum()
        rec = tp / hard_gt.sum()
        meanf = 1.3 * pre * rec / (0.3 * pre + rec)

    # PR curve
    