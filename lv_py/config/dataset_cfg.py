import argparse

def MY77():

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

    return parser


def laboratory():

    parser = argparse.ArgumentParser()

    # path for pascal VOC 2012 dataloader
    parser.add_argument("--image_directory", type=str, default=r"D:\dataset\VOCdevkit\VOC2012\JPEGImages",
                        help="image root which contains JPEG image for dataset")
    parser.add_argument("--semantic_directory", type=str,
                        default=r"D:\dataset\VOCdevkit\VOC2012\SegmentationClass")
    parser.add_argument("--train_txt", type=str,
                        default=r"D:\dataset\VOCdevkit\VOC2012\scribble_train.txt")
    parser.add_argument("--val_txt", type=str,
                        default=r"D:\dataset\VOCdevkit\VOC2012\scribble_val.txt")
    parser.add_argument("--image_level_npy", type=str, 
                        default=r"D:\dataset\VOCdevkit\VOC2012\cls_labels.npy")
    parser.add_argument("--saliency_directory", type=str, default=r"D:\dataset\VOCdevkit\VOC2012\pascal_VOC_Saliency")

    return parser


def MY19():

    parser = argparse.ArgumentParser()
    
    # path for pascal VOC 2012 dataloader
    parser.add_argument("--image_directory", type=str, default="/mnt/diskg/zeyu_yan/VOC2012/VOCdevkit/VOC2012/JPEGImages",
                        help="image root which contains JPEG image for dataset")
    parser.add_argument("--semantic_directory", type=str,
                        default="/mnt/diskg/zeyu_yan/VOC2012/VOCdevkit/VOC2012/SegmentationClassAug")
    parser.add_argument("--train_txt", type=str,
                        default="/home/zeyu_yan/VOC2012/scribble_train.txt")
    parser.add_argument("--val_txt", type=str,
                        default="/mnt/diskg/zeyu_yan/VOC2012/VOCdevkit/VOC2012/scribble_val.txt")
    parser.add_argument("--image_level_npy", type=str, 
                        default="/mnt/diskg/zeyu_yan/VOC2012/cls_labels.npy")
    parser.add_argument("--saliency_directory", type=str, default="/home/zeyu_yan/saliency_mask")

    return parser