"""
    2020/12/01 - now

    This file provide the parameters for initializing your model

"""

# the size of image
RESIZE = (224, 224)

# batch_size of data_loader
BATCH_SIZE = 32

# num_work of data_loader
NUM_WORKERS = 1


"""
    class torchvision.transform.ToTensor
    PIL.Image or numpy.nd-array data whose pixel value range of shape=(H,W,C) is [0, 255]
    converted into pixel data of shape=(C,H,W) and normalized to the torch.FloatTensor type of [0.0, 1.0].

    ImageNet:
        channel =     R      G      B     
        mean    =  [0.485, 0.456, 0.406]
        std     =  [0.229, 0.224, 0.225]
    the average and standard variance for RGB channels pretrained from ImageNet
"""
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]