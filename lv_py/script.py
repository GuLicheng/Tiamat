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
from utils.metric import Cls_Accuracy
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


def test_after_one_epoch(args, rank, epoch, dataloader, model: nn.Module, device, logger):

    if rank != 0:
        return

    tbar = tqdm(dataloader, dynamic_ncols=True)

    evaluator = Cls_Accuracy()

    for idx, sample in enumerate(tbar):

        image, target = sample["image"].to(device), sample["class"].to(device)
        with torch.no_grad():
            logits = model(image)

        evaluator.update(logit=logits, label=target)

    logger.info(f"Testing: Epoch: {epoch}/{args.epochs}, Acc = {evaluator.compute_avg_acc():.4f}")
    # torch.save(model.state_dict(), f"V3{epoch}.pth")

def main(rank, args):


    # frozen seed
    seed = args.seed + rank
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.set_device(rank)

    is_distributed = len(args.gpus) > 1
    if is_distributed:
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus), rank=rank)

    # check device
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = rank

    # build logger 
    logger = Logger(f"{args.work_dir}")

    # write config in main process
    for k, v in vars(args).items():
        logger.info(f"{k} = {v}")
    logger.info("=====================================================================================")

    # build dataste
    trainset = PascalVoc(
        img_dir=args.image_directory, 
        anno_dir=args.semantic_directory, 
        split=args.train_txt, 
        class_label=args.image_level_npy, 
        pipeline=PascalVoc.IMAGE_LEVEL_TRAIN)

    valset = PascalVoc(
        img_dir=args.image_directory, 
        anno_dir=args.semantic_directory, 
        split=args.val_txt, 
        class_label=args.image_level_npy, 
        pipeline=PascalVoc.IMAGE_LEVEL_VAL)

    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(trainset)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, drop_last=True, pin_memory=True, num_workers=args.num_workers,sampler=train_sampler)
    else:
        trainloader = DataLoader(trainset, batch_size=args.batch_size, drop_last=True, pin_memory=True, num_workers=args.num_workers)
    
    testloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=0)

    # build model
    import backbones.vittmp
    backbone = backbones.vittmp.vit_small()
    dino = "/home/zeyu_yan/.cache/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth"
    imagenet224 = "/home/zeyu_yan/.cache/torch/hub/checkpoints/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    backbone.init_params(imagenet224)
    classifier = nn.Conv2d(384, PascalVoc.NUM_CLASSES - 1, 1)
    # classifier = nn.Linear(384, PascalVoc.NUM_CLASSES - 1)
    # model = nn.Sequential(backbone, classifier)
    # model = model.to(rank)

    class Classifier(nn.Module):

        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier

        def forward(self, x):

            vit_outputs = self.backbone(x)

            x = self.classifier(vit_outputs) # change channel-dim
            
            logit = self.fc(x)

            return logit

        def fc(self, x):
            x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
            x = x.view(-1, 20)
            return x

    model = Classifier(backbone, classifier)
    model = model.to(rank)
    train_params = [
        # {'params': backbone.parameters(), 'lr': args.lr},
        {'params': classifier.parameters(), 'lr': args.lr * 10},
    ]


    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])


    # build optimizer and scheduler
    optimizer = torch.optim.AdamW(
            params=train_params,
            lr=args.lr, # linear scaling rule
            # momentum=args.momentum,
            weight_decay=args.weight_decay, # we do not apply weight decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9)

    for epoch in range(args.epochs):
        
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        train_one_epoch(args, rank, epoch, trainloader, model, device, optimizer, scheduler, logger)
        test_after_one_epoch(args, rank, epoch, testloader, model, device, logger)

    if is_distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    

    # path for pascal VOC 2012 dataloader
    parser.add_argument("--image_directory", type=str, default="/mnt/diskg/zeyu_yan/VOC2012/VOCdevkit/VOC2012/JPEGImages",
                        help="image root which contains JPEG image for dataset")
    parser.add_argument("--semantic_directory", type=str,
                        default="/mnt/diskg/zeyu_yan/VOC2012/VOCdevkit/VOC2012/SegmentationClassAug")
    parser.add_argument("--train_txt", type=str,
                        # default="/mnt/diskg/zeyu_yan/VOC2012/VOCdevkit/VOC2012/scribble_train.txt")
                        default="/home/zeyu_yan/low_shot20V2.txt")
    parser.add_argument("--val_txt", type=str,
                        default="/mnt/diskg/zeyu_yan/VOC2012/VOCdevkit/VOC2012/scribble_val.txt")
    parser.add_argument("--image_level_npy", type=str, 
                        default="/mnt/diskg/zeyu_yan/VOC2012/cls_labels.npy")

    # parser.add_argument('--gpus', type=str, default="[0, 1]", type=eval)
    parser.add_argument('--cuda_visible_devices', type=str, default="0")
    parser.add_argument("--work_dir", default="work_dir_tmp", type=str)
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="epoch num")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
    parser.add_argument("--seed", type=int, default=17, help="seed for random number")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum')
    parser.add_argument('--rank', type=float, default=0)
    parser.add_argument('--pretrained', type=str, default="/mnt/diskg/zeyu_yan/mit_b2.pth")

    args = parser.parse_args()

    # calculate gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.gpus = list(range(len(args.cuda_visible_devices.split(','))))

    world_size = len(args.gpus)

    if world_size == 1:
        main(0, args)

    else:
        port = str(random.randint(1025, 65535))
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = port
        mp.spawn(main, args=[args], nprocs=len(args.gpus), join=True)


