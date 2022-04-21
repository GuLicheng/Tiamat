from copy import deepcopy
from email.policy import default
from functools import reduce

from tqdm import tqdm
from dataset.pascalvoc import PascalVoc
from config import dataset_cfg
import argparse
import torch.nn as nn
from torchvision.transforms import transforms
from dataset.pipeline import *
from torch.utils.data import DataLoader
from utils.metric import Cls_Accuracy, Evaluator
from utils.logger import AverageMeter, Logger
import torch.nn.functional as F
import cv2 as cv



def train_one_epoch(args, epoch, rank, dataloader, model, device, optimizer, logger):
    
    model = model.train().to(device)

    train_loss1, train_loss2, train_loss = AverageMeter(), AverageMeter(), AverageMeter()

    loss, loss1, loss2 = 0., 0., 0.

    for idx, sample in enumerate(dataloader):

        image, target = \
            sample["image"].to(device), sample["semantic"].to(device)

        attention_mask = model(image)["attention_mask"]

        B, HEAD, C, HW = attention_mask.shape # [1, 6, 20, 1024?]

        attention_mask = attention_mask.view(B * HEAD, C, int(HW ** 0.5), -1)

        # [6, 20, 512, 512]
        attention_mask = F.interpolate(attention_mask, size=target.shape[-2:], mode="bilinear", align_corners=False)


        # make label for neg
        # mask = torch.zeros(size=(C, ) + target.shape[-2:], device=image.device, dtype=image.dtype)

        # make label for pos
        mask = torch.where(target == idx + 1, 1, 0).to(device).float().detach()


        # train_loss1.update(loss1.item())
        # train_loss2.update(loss2.item())

        # loss = loss1 + loss2

        # optimizer.zero_grad()

        # train_loss.update(loss.item())

        # loss.backward()

        # optimizer.step()

        # logger.info(f"Epoch: {epoch}/{args.epochs}")
        # logger.info(f"Loss: {train_loss.avg:.4f} = loss1: {train_loss1.avg:.4f} + loss2: {train_loss2.avg:.4f}")


def visual_attention_mask(args, epoch, rank, dataloader, model, device, logger):

    model = model.eval().to(device)

    image_list = []

    for idx, sample in enumerate(dataloader):

        image, class_vector = sample["image"].to(device), sample["class"].to(device)

        attention_mask = model(image)["attention_mask"] # [1, 6, C, HW]

        B, HEAD, C, HW = attention_mask.shape # [1, 6, 20, 1024?]

        attention_mask = attention_mask.view(B * HEAD, C, int(HW ** 0.5), -1)

        attention_mask = F.interpolate(attention_mask, size=(224, 224), mode="bilinear", align_corners=False) # [6, 20, 224, 224]

        attention_mask = attention_mask[:, idx, ...] # [6, 224, 224]

        attention_mask = attention_mask.sigmoid() * 255
        
        attention_mask = attention_mask.detach().cpu().numpy().astype(np.uint8) 

        image_list.append(
            np.hstack([attention_mask[h] for h in range(HEAD)])
        )

    cv.imwrite(f"Mask{epoch}.png", np.vstack(image_list))


def visual_all_class_token_attention_mask(args, epoch, rank, dataloader, model, device, logger, sample_name):
    
    model = model.eval().to(device)

    sample = dataloader.dataset[sample_name]

    image = sample["image"].to(device)

    attention_mask = model(image[None, ...])["attention_mask"] # [1, 6, C, HW]

    B, HEAD, C, HW = attention_mask.shape # [1, 6, 20, 1024?]

    attention_mask = attention_mask.view(B * HEAD, C, int(HW ** 0.5), -1)

    attention_mask = F.interpolate(attention_mask, size=(224, 224), mode="bilinear", align_corners=False) # [6, 20, 224, 224]

    attention_mask = attention_mask.permute(0, 2, 1, 3).contiguous().view(6 * 224, -1)

    attention_mask = (attention_mask.sigmoid().detach().cpu().numpy() * 255).astype(np.uint8)

    cv.imwrite(f"Mask_All_Class{epoch}.png", attention_mask)


def main():

    parser = dataset_cfg.MY19()

    parser.add_argument("--lowshot_txt", default="/home/zeyu_yan/20.txt")
    parser.add_argument("--dino_pretrain", default="/home/zeyu_yan/.cache/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth")
    parser.add_argument("--imagenet_pretrain", 
        default="/home/zeyu_yan/.cache/torch/hub/checkpoints/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz")

    parser.add_argument("--work_dir", default="work_dir_tmp", type=str)
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--epochs", type=int, default=500, help="epoch num")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
    parser.add_argument("--num_workers", type=int, default=0, help="num_workers")
    parser.add_argument("--seed", type=int, default=17, help="seed for random number")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum')
    parser.add_argument('--rank', type=float, default=0)

    args = parser.parse_args()

    pipeline = transforms.Compose([
        ReadImage(),
        ToTensor(args=["image"]),
        Resize(args=["image"], size=(512, 512)),
        NormalizeImage(),
    ])

    # for val
    valset = PascalVoc(
        img_dir=args.image_directory, 
        anno_dir=args.semantic_directory, 
        split=args.lowshot_txt, 
        class_label=args.image_level_npy, 
        pipeline=pipeline)

    testloader = DataLoader(valset, batch_size=1, pin_memory=False, num_workers=0)

    trainset = PascalVoc(
        img_dir=args.image_directory, 
        anno_dir=args.semantic_directory, 
        split=args.lowshot_txt, 
        class_label=args.image_level_npy, 
        pipeline=PascalVoc.SEMANTIC_TRAIN)
    trainloader = DataLoader(trainset, batch_size=1, drop_last=False, pin_memory=True, num_workers=0)


    # for visual all class attention mask

    valset2 = PascalVoc(
        img_dir=args.image_directory, 
        anno_dir=args.semantic_directory, 
        sal_dir=args.saliency_directory,
        split=args.val_txt, 
        class_label=args.image_level_npy, 
        pipeline=pipeline
    )

    testloader2 = DataLoader(valset2, batch_size=1, pin_memory=False, num_workers=0)

    # build model
    from backbones.dino_vit import vit_small
    backbone = vit_small(num_cls_tokens=20)
    backbone.init_params(args.dino_pretrain)

    # build optimizer

    train_params = [
        {'params': backbone.cls_token, 'lr': args.lr},
    ]
    optimizer = torch.optim.AdamW(
        params=train_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # build logger
    logger = Logger(args.work_dir, use_stdout=True)

    for epoch in range(args.epochs):
        
        train_one_epoch(args, epoch, 0, trainloader, backbone, 0, optimizer, logger)
        
        if epoch % 50 == 0:
            visual_attention_mask(args, epoch, 0, testloader, backbone, 0, logger)
            visual_all_class_token_attention_mask(args, epoch, 0, testloader2, backbone, 0, logger, "2007_000129")


if __name__ == "__main__":
    main()


