import numpy as np
import torch


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return IoU


    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class Cls_Accuracy():
    def __init__(self, threshold = 0.5):
        self.total = 0
        self.correct = 0
        self.threshold = threshold
        

    def update(self, logit, label):
        
        logit = logit.sigmoid()
        logit = (logit >= self.threshold)
        all_correct = torch.all(logit == label.byte(), dim=1).float().sum().item()
        
        self.total += logit.size(0)
        self.correct += all_correct


    def compute_avg_acc(self):
        return self.correct / self.total

    def reset(self):
        self.total = 0
        self.correct = 0


class BER_Evaluate:

    def __init__(self, threshold = 0.5) -> None:
        self.threshold = threshold
        self.TP,self. TN, self.POS, self.NEG = 0, 0, 0, 0
        self.FP, self.FN = 0, 0

    def update(self, gt, pred):

        assert 0 <= gt.max() <= 1 and 0 <= pred.max() <= 1

        posPoints = gt > self.threshold
        negPoints = gt <= self.threshold

        countPos = posPoints.sum()
        countNeg = negPoints.sum()

        tp = posPoints & (pred > self.threshold)
        countTP = tp.sum()

        tn = negPoints & (pred <= self.threshold)
        countTN = tn.sum()


        self.FP += ((posPoints) & (pred > self.threshold)).sum()
        self.FN += ((negPoints) & (pred < self.threshold)).sum()

        self.TP += countTP
        self.TN += countTN
        self.POS += countPos
        self.NEG += countNeg

    def calculate(self):

        posAcc = self.TP / self.POS
        negAcc = self.TN / self.NEG

        BER = 0.5 * (2 - posAcc - negAcc)

        acc_final = (self.TP + self.TN) / (self.POS + self.NEG)
        final_BER = BER * 100
        pErr = (1 - posAcc) * 100
        nErr = (1 - negAcc) * 100

        print(f"BER: {final_BER:.2f}, pErr: {pErr:.2f}, nErr: {nErr:.2f}, acc: {acc_final:.4f}")

        return acc_final, final_BER, pErr, nErr