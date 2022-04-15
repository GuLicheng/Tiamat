from copy import deepcopy
from email.policy import default

from tqdm import tqdm
from dataset.pascalvoc import PascalVoc
from config import dataset_cfg
import argparse
import torch.nn as nn
from torchvision.transforms import transforms
from dataset.pipeline import *
from torch.utils.data import DataLoader
from utils.metric import Cls_Accuracy, Evaluator
from utils.logger import Logger
import torch.nn.functional as F
import cv2 as cv


def cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    sim_matrix = tensor1.matmul(tensor2.transpose(-2, -1))
    a = torch.norm(tensor1, p=2, dim=-1)
    b = torch.norm(tensor2, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    sim_matrix.nan_to_num_(0)
    return sim_matrix

class Classifier(nn.Module):

    def __init__(self, backbone, classifier: nn.Conv2d):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forwardV2(self, x, return_cam = False):

        vit_outputs, class_token = self.backbone.forward_with_patch_and_class_token(x)

        B, C, H, W = vit_outputs.shape

        x1 = vit_outputs.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x2 = self.classifier.weight.view(20, 384)

        x = cosine_similarity(x1, x2).permute(0, 2, 1).view(B, -1, H, W)

        # logit = self.fc(x)
        logit = cosine_similarity(class_token, x2)

        if return_cam:
            return {
                "cam": x,
                "pred_cls": logit,
            }


        return logit

    def forward(self, x, return_cam = False):
    
        vit_outputs = self.backbone(x)

        # x = self.classifier(vit_outputs) # change channel-dim

        B, C, H, W = vit_outputs.shape

        x1 = vit_outputs.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x2 = self.classifier.weight.view(20, 384)

        x = cosine_similarity(x1, x2).permute(0, 2, 1).view(B, -1, H, W)

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


def simple_test(args):
    pass

def min_max_norm(tensor, min_val = 0, max_val = 1):
    min_v, max_v = tensor.min(), tensor.max()
    tensor = (tensor - min_v) / (max_v - min_v + 1e-5) * (max_val - min_val) + min_val
    return tensor 

def test_after_one_epoch(args, rank, dataloader, model: Classifier, device, logger):
    
    # device = torch.device("cpu")

    tbar = tqdm(dataloader, dynamic_ncols=True)
    model = model.eval().to(device)

    # for compare

    iou_e = Evaluator(21)

    def multi_scale_prediction(scale, model, image):

        input_tensor = F.interpolate(image, size=(scale, scale), mode="bilinear", align_corners=False)

        result = model(input_tensor)

        return result


    for idx, sample in enumerate(tbar):

        image, target, sal, gt = \
            sample["image"].to(device), sample["class"].to(device), sample["saliency"].to(device), sample["semantic"]
        with torch.no_grad():
            result = model(image, return_cam = True)


        logits = result["pred_cls"]
        pred = logits.detach()



        cam = result["cam"]
        cam = model.cam_normalize(cam, image.shape[-2:], pred)[0].argmax(dim=0) + 1

        name = sample["ori_name"][0] 

        # cv.imwrite(f"CAM/{name}.png", cam.cpu().detach().numpy().astype(np.uint8)) # just save cam

        cam[sal[0] == 0] = 0
        
        pre_image = cam.cpu().detach().numpy()

        gt_image = sample["semantic"][0].long().numpy()
        iou_e.add_batch(pre_image=pre_image, gt_image=gt_image)

    if logger is not None:
        logger.info(f"mIoU = {iou_e.Mean_Intersection_over_Union()}")
    else:
        print("mIoU = ", iou_e.Mean_Intersection_over_Union())

    for k, v in zip(PascalVoc.CAT_LIST_SCRIBBLE, iou_e.Intersection_over_Union()):
        print(f"{k} = {v}")



def make_prototype(trainset, backbone):
    cls_tokens = []
    backbone.cuda()
    for sample in trainset:

        image = sample["image"]
        image = image[None, :, :, :].cuda()

        with torch.no_grad():
            cls_token = backbone.forward_backup(image)

        cls_tokens.append(cls_token)

    weights = torch.cat(cls_tokens, dim=0)

    return weights

def main():

    parser = dataset_cfg.MY77()

    parser.add_argument("--lowshot_txt", default="/home/zeyu_yan/20.txt")
    parser.add_argument("--dino_pretrain", default="/home/zeyu_yan/.cache/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth")
    parser.add_argument("--imagenet_pretrain", 
        default="/home/zeyu_yan/.cache/torch/hub/checkpoints/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz")

    args = parser.parse_args()

    pipeline = transforms.Compose([
        ReadImage(),
        ReadAnnotation(key="saliency"),
        ReadAnnotation(key="semantic"),
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

    trainset = PascalVoc(
        img_dir=args.image_directory, 
        anno_dir=args.semantic_directory, 
        split=args.lowshot_txt, 
        class_label=args.image_level_npy, 
        pipeline=PascalVoc.IMAGE_LEVEL_VAL)
    # trainloader = DataLoader(trainset, batch_size=1, drop_last=False, pin_memory=True, num_workers=0)

    # build model
    import backbones.vittmp
    backbone = backbones.vittmp.vit_small()
    backbone.init_params(args.imagenet_pretrain)
    classifier = nn.Conv2d(384, PascalVoc.NUM_CLASSES - 1, 1, bias=False)

    candidate = reversed([
"2008_001610",
"2010_001418",
"2008_003562",
"2010_002973",
"2008_007239",
"2009_005236",
"2011_001135",
"2010_003887",
"2011_000379",
"2008_001876",
"2010_004963",
"2007_007530",
"2010_001457",
"2008_007472",
"2009_001444",
"2009_000535",
"2009_000684",
"2008_006843",
"2010_001842",
"2007_000876",
"2010_005796",
"2010_002286",
"2010_000114",
"2011_002389",
"2010_005505",
"2009_002972",
"2010_005317",
"2010_005202",
"2009_001502",
"2007_007878",
"2010_002811",
"2010_003230",
"2007_009630",
"2009_004887",
"2011_003216",
"2011_000108",
"2010_001347",
"2008_005716",
"2010_001247",
"2010_000815",
"2009_003660",
"2010_005791",
"2009_003697",
"2011_000345",
"2010_002778",
"2010_005128",
"2007_008575",
"2010_004144",
"2010_001939",
"2007_009464",
"2010_002573",
"2007_003525",
"2007_007098",
"2010_004540",
"2008_001741",
"2010_004306",
"2009_005160",
"2010_001590",
"2008_004583",
"2009_001117",
"2010_003696",
"2008_002215",
"2010_000043",
"2010_001386",
"2009_004705",
"2010_000675",
"2007_006641",
"2011_000999",
"2008_005600",
"2008_005300",
"2010_002625",
"2010_004060",
"2008_002749",
"2010_005627",
"2009_003921",
"2009_000553",
"2010_003974",
"2009_002917",
"2008_002182",
"2008_007165",
"2010_004760",
"2010_003174",
"2010_004933",
"2011_001754",
"2010_001177",
"2010_000661",
"2010_005232",
"2008_000860",
"2010_003665",
"2010_000469",
"2007_008403",
"2007_006303",
"2007_009724",
"2009_004434",
"2007_003778",
"2008_002177",
"2007_004998",
"2010_001660",
"2009_001205",
"2010_004365",
"2010_005513",
"2009_002628",
"2009_001270",
"2011_000973",
"2010_000404",
"2011_000469",
"2008_001592",
"2010_000269",
"2007_005688",
"2010_004808",
"2007_001825",
"2007_000549",
"2011_000758",
"2009_005037",
"2007_003788",
"2009_001894",
"2010_003925",
"2008_005770",
"2010_003950",
"2010_000519",
"2009_003783",
"2007_002760",
"2008_002067",
"2007_000528",
"2009_005177",
"2010_005805",
"2007_009216",
"2010_005652",
"2007_001185",
"2010_001195",
"2010_004171",

    ])

    # prototype
    weights = make_prototype(trainset, backbone)

    weights = nn.parameter.Parameter(weights.detach().clone())[:, :, None, None]
    classifier.weight.data = weights

    model = Classifier(backbone, classifier)
    model = model.eval().cuda()
    test_after_one_epoch(args, 0, testloader, model, 0, None)



if __name__ == "__main__":
    main()


