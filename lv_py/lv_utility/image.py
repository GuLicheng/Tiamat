import numpy as np
import cv2 as cv
import os
import functools

class ImageHandle:

    size = (224, 224)

    @staticmethod
    def files(root: str):
        root = root.replace('\\', '/')
        return [ImageHandle.path_join(root, filename) for filename in os.listdir(root)]

    @staticmethod
    def path_join(*args):
        return functools.reduce(lambda x, y: x + '/' + y, args)

    @staticmethod
    def write_branch(branches: dict, dest_root: str, img_name: str, dsize = size):
        """
            @param:
                dict: pair of branch name and list of tensors such as  
                    {"branch1": [tensor1, tensor2, ...], "branch2": [tensor1, ...]}
                dest_root: such as following
                img_name: such as following
                dsize: the picture size

            dest_root
                └──  img_name
                        ├── branch1.png
                        │
                        └── branch2.png
        """
        dest_root = dest_root.replace('\\', '/')
        if not os.path.exists(dest_root):
            os.makedirs(dest_root)
        img_dirname = ImageHandle.path_join(dest_root, img_name)
        if not os.path.exists(img_dirname):
            os.makedirs(img_dirname)
        
        for key, value in branches.items():
            for idx, tensor in enumerate(value):
                img = tensor.squeeze().cpu().detach().numpy()
                img = cv.resize(img, dsize=dsize)
                dir_name = ImageHandle.path_join(img_dirname, f"{key}_{idx}.png")
                cv.imwrite(dir_name, img * 255)
                print(dir_name)
        
    @staticmethod
    def show_group(root: str, name = "", dsize = size):
        """
            @param:
                root: dest_root/img_name (see `write_branch`)
                name: windowname for opencv
                dsize: the picture size
        """
        if not os.path.exists(root):
            raise FileExistsError()
        def transform(path, dsize):
            print(path)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, dsize=dsize)
            return img

        def stack_image(imgs):
            length = imgs.__len__()
            if length % 3 == 0:
                h1 = np.hstack(imgs[: length // 3])
                h2 = np.hstack(imgs[length // 3: length // 3 * 2])
                h3 = np.hstack(imgs[length // 3 * 2:])
                return np.vstack([h1, h2, h3]), 300 * length, 900
            elif length % 2 == 0:
                h1 = np.hstack(imgs[: length // 2])
                h2 = np.hstack(imgs[length // 2:])
                return np.vstack([h1, h2]), 300 * length, 600
            else:
                return np.hstack(imgs), 300 * length, 300

        imgs = [transform(img, dsize=dsize) for img in ImageHandle.files(root)]
        imgs, w, h = stack_image(imgs)
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.resizeWindow(name, w, h)
        cv.imshow(name, imgs)
        cv.waitKey(0)
        
    @staticmethod
    def HSV(filename):
       pass

