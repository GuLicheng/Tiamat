import os
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from dataset.saliency import SBU
import argparse
import cv2 as cv
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.logger import Logger, AverageMeter
from utils.metric import BER_Evaluate
import torch.multiprocessing as mp

def train_one_epoch(args, rank, epoch, dataloader, model, device, optimizer, scheduler, logger):

    torch.cuda.empty_cache()
    
    model = model.to(device)

    train_loss = AverageMeter("BCELoss")

    tbar = tqdm(dataloader, dynamic_ncols=True)

    for idx, sample in enumerate(tbar):

        optimizer.zero_grad()

        image, target = sample["image"].to(device), sample["saliency"].to(device)

        logits = model(image)

        target /= 255

        loss = F.binary_cross_entropy_with_logits(logits, target[:, None, :, :].detach())

        loss.backward()

        optimizer.step()
        scheduler.step()

        train_loss.update(loss.item())

        tbar.set_description(f"{train_loss.name}: {train_loss.avg:.4f}")

    logger.info(f"Training: Epoch: {epoch}/{args.epochs}")


def test_after_one_epoch(args, rank, epoch, dataloader, model, device, logger):

    tbar = tqdm(dataloader, dynamic_ncols=True)
    model = model.eval()

    evaluator = BER_Evaluate()

    for idx, sample in enumerate(tbar):

        image, target = sample["image"].to(device), sample["saliency"].to(device)
        with torch.no_grad():
            logits = model(image)
        evaluator.update(logit=logits.sigmoid().detach().cpu().numpy(), 
            label=(target[:, None, :, :]/255).detach().cpu().numpy())

    acc_final, final_BER, pErr, nErr = evaluator.calculate_ber()
    logger.info(f"Testing: BER = {final_BER:.2f}")
    torch.save(model.state_dict(), f"{args.work_dir}/{epoch}.pth")

def single_gpu_test(args):

    from decode_head.segformer_head import mitb4_model
    model = mitb4_model(None, None) 
    ckpt = torch.load("/home/zeyu_yan/work_dir_tmp/25.pth", map_location="cpu")
    def remove_module_prefix(params):
        
        def transform(kv):
            k, v = kv
            assert k.startswith("module.")
            return k[7:], v

        return dict(map(transform, params.items()))

    ckpt = remove_module_prefix(ckpt)

    model.load_state_dict(ckpt)

    model = model.eval().cuda()

    dataset = SBU(args.test_image_root, pipeline=SBU.SALIENCY_VAL)

    e = BER_Evaluate()

    for sample in tqdm(dataset):

        image, gt = sample['image'].cuda(), sample["saliency"].cuda()

        pred = model(image[None, :, :, :])

        e.update(pred.sigmoid(), gt / 255)

    acc_final, final_BER, pErr, nErr = e.calculate_ber()
    print(final_BER)
    exit()


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

    trainset = SBU(args.train_image_root, pipeline=SBU.SALIENCY_TRAIN)
    valset = SBU(args.test_image_root, pipeline=SBU.SALIENCY_VAL)

    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(trainset)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, drop_last=True, pin_memory=True, num_workers=args.num_workers,sampler=train_sampler)
    else:
        trainloader = DataLoader(trainset, batch_size=args.batch_size, drop_last=True, pin_memory=True, num_workers=args.num_workers)
    
    testloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=0)

    # build model
    from decode_head.segformer_head import mitb4_model
    model = mitb4_model(args.pretrained, None) 

    model = model.to(rank)
    train_params = [
        {'params': model.get_1x_lr_params(), 'lr': args.lr},
        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10},
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

    from utils.poly_scheduler import PolynomialLRDecay
    scheduler = PolynomialLRDecay(optimizer=optimizer, max_decay_steps=args.epochs, end_learning_rate=0, power=0.9)

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
    parser.add_argument("--train_image_root", type=str, default="/mnt/diskg/zeyu_yan/SBUshadow/SBUTrain",
                        help="image root which contains JPEG image for dataset")
    parser.add_argument("--test_image_root", type=str, default="/mnt/diskg/zeyu_yan/SBUshadow/SBUTestRename",
                        help="image root which contains JPEG image for dataset")

    # parser.add_argument('--gpus', type=str, default="[0, 1]", type=eval)
    parser.add_argument('--cuda_visible_devices', type=str, default="0,1,2,3")
    parser.add_argument("--work_dir", default="work_dir_tmp", type=str)
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--epochs", type=int, default=30, help="epoch num")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
    parser.add_argument("--num_workers", type=int, default=0, help="num_workers")
    parser.add_argument("--seed", type=int, default=17, help="seed for random number")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum')
    parser.add_argument('--rank', type=float, default=0)
    parser.add_argument('--pretrained', type=str, default="/mnt/diskg/zeyu_yan/mit_b4.pth")
    parser.add_argument('--resume', type=str, default="/home/zeyu_yan/work_dir_tmp/25.pth")

    args = parser.parse_args()

    # calculate gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.gpus = list(range(len(args.cuda_visible_devices.split(','))))

    world_size = len(args.gpus)

    if world_size == 1:
        main(0, args)

    else:
        mp.set_start_method("spawn")
        port = str(random.randint(1025, 65535))
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = port
        mp.spawn(main, args=[args], nprocs=len(args.gpus), join=True)


