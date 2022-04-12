from math import dist
import os
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from dataset.pascalvoc import PascalVoc
import argparse
import cv2 as cv
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.logger import Logger, AverageMeter
from utils.metric import Cls_Accuracy, Evaluator
from backbones.vit import vit_small
from backbones.segformer import mit_b2
import torch.multiprocessing as mp

def train_one_epoch(args, rank, epoch, dataloader, model, device, optimizer, scheduler, logger):

    torch.cuda.empty_cache()
    
    model = model.to(device)

    loss1 = AverageMeter("multilabel_soft_margin_loss")
    train_loss = AverageMeter("Train Loss")

    tbar = tqdm(dataloader, dynamic_ncols=True)

    for idx, sample in enumerate(tbar):

        optimizer.zero_grad()

        image, target = sample["image"].to(device), sample["class"].to(device)

        logits = model(image)

        loss = F.multilabel_soft_margin_loss(logits, target)

        loss.backward()

        optimizer.step()
        scheduler.step()

        loss1.update(loss.item())

        train_loss.update(loss.item())

        tbar.set_description(f"{train_loss.name}: {train_loss.avg:.4f}, {loss1.name}: {loss1.avg:.4f}")

    logger.info(f"Training: Epoch: {epoch}/{args.epochs}")
    logger.info('Loss: %.4f' % loss1.get_avg())


def test_after_one_epoch(args, rank, dataloader, model: nn.Module, device, logger):

    tbar = tqdm(dataloader, dynamic_ncols=True)
    model = model.eval().to(device)
    evaluator = Cls_Accuracy()

    iou_e = Evaluator(21)

    for idx, sample in enumerate(tbar):

        image, target, sal, gt = \
            sample["image"].to(device), sample["class"].to(device), sample["saliency"].to(device), sample["semantic"]
        with torch.no_grad():
            result = model(image, return_cam = True)

        logits = result["pred_cls"]
        cam = result["cam"]

        evaluator.update(logit=logits, label=target)

        pred = logits.detach().sigmoid()

        cam = model.cam_normalize(cam, image.shape[-2:], pred)[0].argmax(dim=0) + 1

        cam[gt[0] == 0] = 0
        
        pre_image = cam.cpu().detach().numpy()
        gt_image = sample["semantic"][0].long().numpy()

        iou_e.add_batch(pre_image=pre_image, gt_image=gt_image)

        if ((logits.detach().sigmoid() > 0.5).long() - target).sum() != 0:
            seg1 = PascalVoc.decode_segmap(pre_image)
            seg2 = PascalVoc.decode_segmap(gt_image)
            seg = np.hstack([seg1, seg2])
            cv.imwrite("seg.png", seg)
            i = 0


        # name = sample["ori_name"][0]

        # cv.imwrite(f"segmap1/{name}.png", cam.cpu().detach().numpy().astype(np.uint8))


    print("acc = ", evaluator.compute_avg_acc())
    print("mIoU = ", iou_e.Mean_Intersection_over_Union())

from dataset.pipeline import (ReadImage, RandomScaleCrop, 
RandomHorizontalFlip, ColorJitterImage, ToTensor, NormalizeImage, ReadAnnonation)
from torchvision.transforms import transforms

def main(args):


    device = 0

    # build logger 
    logger = Logger(f"{args.work_dir}")


    pipeline = transforms.Compose([
        ReadImage(),
        ReadAnnonation(key="saliency"),
        ReadAnnonation(key="semantic"),
        ToTensor(args=["image", "saliency", "semantic"]),
        NormalizeImage(),
    ])

    valset = PascalVoc(
        img_dir=args.image_directory, 
        anno_dir=args.semantic_directory, 
        sal_dir=args.saliency_directory,
        split=args.val_txt, 
        class_label=args.image_level_npy, 
        pipeline=pipeline)

    testloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=0)

    # build model
    import backbones.vittmp
    backbone = backbones.vittmp.vit_small()
    classifier = nn.Conv2d(384, PascalVoc.NUM_CLASSES - 1, 1)

    class Classifier(nn.Module):

        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier

        def forward(self, x, return_cam = False):

            vit_outputs = self.backbone(x)

            x = self.classifier(vit_outputs) # change channel-dim
            
            logit = self.fc(x)

            if return_cam:
                return {
                    "cam": x,
                    "pred_cls": logit,
                }


            return logit

        def fc(self, x):
            x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
            x = x.view(-1, 20)
            return x

        def cam_normalize(self, cam, size, label):
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=False)
            cam /= F.adaptive_max_pool2d(cam, 1) + 1e-5
            
            cam = cam * label[:, :, None, None] # clean
            
            return cam


    model = Classifier(backbone, classifier)

    def remove_module_prefix(params):
        
        def transform(kv):
            k, v = kv
            assert k.startswith("module.")
            return k[7:], v

        return dict(map(transform, params.items()))

    ckpt = torch.load("/home/zeyu_yan/V27.pth", map_location="cpu")
    model.load_state_dict((remove_module_prefix(ckpt)))

    test_after_one_epoch(args, 0, testloader, model, device, logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    

    # path for pascal VOC 2012 dataloader
    parser.add_argument("--image_directory", type=str, default="/home/zeyu_yan/VOC2012/JPEGImages",
                        help="image root which contains JPEG image for dataset")
    parser.add_argument("--semantic_directory", type=str,
                        default="/home/zeyu_yan/VOC2012/SegmentationClass")
    parser.add_argument("--train_txt", type=str,
                        default="/home/zeyu_yan/VOC2012/scribble_train.txt")
    parser.add_argument("--val_txt", type=str,
                        default="/home/zeyu_yan/VOC2012/val.txt")
    parser.add_argument("--image_level_npy", type=str, 
                        default="/home/zeyu_yan/VOC2012/cls_labels.npy")
    parser.add_argument("--saliency_directory", type=str, default="/home/zeyu_yan/saliency_mask")

    # parser.add_argument('--gpus', type=str, default="[0, 1]", type=eval)
    parser.add_argument('--cuda_visible_devices', type=str, default="0,1,2,3")
    parser.add_argument("--work_dir", default="work_dir_tmp", type=str)
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="epoch num")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
    parser.add_argument("--seed", type=int, default=17, help="seed for random number")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum')
    parser.add_argument('--rank', type=float, default=0)


    args = parser.parse_args()

    # calculate gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.gpus = list(range(len(args.cuda_visible_devices.split(','))))

    world_size = len(args.gpus)

    main(args)



