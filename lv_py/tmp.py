from functools import reduce
import itertools
from tqdm import tqdm
from dataset.pascalvoc import PascalVoc
from config import dataset_cfg
import argparse
import torch.nn as nn
from torchvision.transforms import transforms
from dataset.pipeline import *
from torch.utils.data import DataLoader
from utils.metric import Cls_Accuracy, Evaluator
from utils.logger import AverageMeter, Logger, AverageMeters
import torch.nn.functional as F
import cv2 as cv

def binary_cross_entropy_with_logits_with_ignore_index(input, target, ignore_index):
    """
        assert (target == ignored_index).sum() != target.numel()
    """
    mask = target != ignore_index

    return F.binary_cross_entropy_with_logits(input[mask], target[mask])


def train_one_epoch(args, epoch, rank, dataloader, model, device, optimizer, logger):
    
    torch.cuda.empty_cache()

    model = model.train().to(device)

    train_loss = AverageMeters("loss", "loss1", "loss2")

    for idx, sample in enumerate(dataloader):

        optimizer.zero_grad()

        image, target, class_vector = \
            sample["image"].to(device), sample["semantic"].to(device), sample["class"].to(device)

        output = model(image)

        attn = output["attention_mask"] # torch.Size([1, 6, 20, 1024])

        B, HEAD, C, HW = attn.shape
        attn = attn.view(B * HEAD, C, int(HW ** 0.5), -1) # [6, 20, 32, 32]
        attn = F.interpolate(attn, size=target.shape[-2:], mode="bilinear", align_corners=False) # [6, 20, 512, 512]

        loss1, loss2 = 0., 0.

        all_zeros = torch.zeros(size=target.shape[-2:], dtype=target.dtype, device=target.device, requires_grad=False)
        target[(0 < target) & (target < 255)] = 1
        for i, j in itertools.product(range(HEAD), range(C)):
            if j == idx:
                loss1 += binary_cross_entropy_with_logits_with_ignore_index(attn[i][j], target[0], 255)
            else:
                loss2 += F.binary_cross_entropy_with_logits(attn[i][j], all_zeros)

        
        loss = loss1 + loss2

        train_loss.update("loss1", loss1.item())
        train_loss.update("loss2", loss2.item())
        train_loss.update("loss", loss.item())
        
        loss.backward()

        optimizer.step()

        train_loss.report(logger)
        


def train_one_epoch_debug(args, epoch, rank, dataloader, model, device, optimizer, logger):

    # classification
    model = model.train().to(device)

    for idx, sample in enumerate(dataloader):

        image, target = sample["image"].to(device), sample["class"].to(device)

        pred = model(image)

        loss = F.multilabel_soft_margin_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(f"{loss.item():.4f}")


def visual_attention_mask(args, epoch, rank, dataloader, model, device, logger):

    model = model.eval().to(device)

    image_list = []

    for idx, sample in enumerate(dataloader):

        image, class_vector = sample["image"].to(device), sample["class"].to(device)

        with torch.no_grad():
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

    with torch.no_grad():
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
    trainloader = DataLoader(trainset, batch_size=1, drop_last=False, pin_memory=False, num_workers=0)


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
        
        # if epoch % 50 == 0:
        #     visual_attention_mask(args, epoch, 0, testloader, backbone, 0, logger)
        #     visual_all_class_token_attention_mask(args, epoch, 0, testloader2, backbone, 0, logger, "2007_000129")


if __name__ == "__main__":
    main()


