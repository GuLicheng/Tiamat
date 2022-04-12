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
import torch.nn.functional as F


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
        # logits = min_max_norm(logits)
        pred = logits.detach()
        cam = result["cam"]



        cam = model.cam_normalize(cam, image.shape[-2:], pred)[0].argmax(dim=0) + 1

        cam[sal[0] == 0] = 0
        
        pre_image = cam.cpu().detach().numpy()

        gt_image = sample["semantic"][0].long().numpy()
        iou_e.add_batch(pre_image=pre_image, gt_image=gt_image)

        # name = sample["ori_name"][0]

        # cv.imwrite(f"segmap1/{name}.png", cam.cpu().detach().numpy().astype(np.uint8))


    print("mIoU = ", iou_e.Mean_Intersection_over_Union())


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


    # prototype
    cls_tokens = []

    for sample in trainset:

        image = sample["image"]
        image = image[None, :, :, :]

        with torch.no_grad():
            cls_token = backbone.forward_backup(image)

        cls_tokens.append(cls_token)

    weights = torch.cat(cls_tokens, dim=0)
    weights = nn.parameter.Parameter(weights.detach().clone())[:, :, None, None]
    classifier.weight.data = weights

    model = Classifier(backbone, classifier)

    test_after_one_epoch(args, 0, testloader, model, 0, None)




if __name__ == "__main__":
    main()


