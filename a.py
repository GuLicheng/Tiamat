import numpy as np
import os
import cv2 as cv

ROOT1 = r"F:/SaliencyMap/testmaps_ini"

# ROOT2 = r"F:/SaliencyMap/RGBD_for_test/DES"
# ROOT2 = r"F:/SaliencyMap/RGBD_for_test/DUT-RGBD"
ROOT2 = r"F:/SaliencyMap/RGBD_for_test/NLPR"

# ROOT2 = r"F:/SaliencyMap/RGBD_for_test/NJU2K"
# ROOT2 = r"F:/SaliencyMap/RGBD_for_test/STERE"  # rgb, depth, gt
# ROOT3 = r"C:\Users\Administrator\Desktop\新建文本文档.txt"
ROOT3 = r"C:\Users\Administrator\Desktop\NLPR.txt"
# ROOT3 = r"C:\Users\Administrator\Desktop\nju2k.txt"
# ROOT3 = r"C:\Users\Administrator\Desktop\des.txt"
# ROOT3 = r"C:\Users\Administrator\Desktop\duts_rgbd.txt"

OUTPUT = r"C:\Users\Administrator\Desktop\Result"

class ImageList:

    def __init__(self) -> None:
        self.dataset_name = "NLPR"
        self.root1 = ROOT1
        self.root2 = ROOT2
        self.image_list = self.init_image_list()
        self.image_name = self.init_image_name()


        self.result_name = []

    def init_directories(self):
        directories = os.listdir(self.root1)
        model_name = []
        for directory in directories:
            if not os.path.isdir(self.root1 + '/' + directory):
                continue
            sub_dir = os.listdir(self.root1 + "/" + directory)
            if self.dataset_name not in sub_dir:
                assert False, "Mot exist"
            model_name.append(directory)
        # print(model_name)
        return model_name

    def init_image_name(self):
        image_name = os.listdir(self.root2 + '/' + "rgb")
        for i in range(len(image_name)):
            image_name[i] = image_name[i].split('.')[0]
        # directories = ["depth", "gt", "rgb"]
        # print(image_name)
        return image_name

    def init_image_list(self):
        model_name = self.init_directories()
        image_list = []
        for dir_name in ["RGB", "GT", "depth"]:
            image_list.append(self.root2 + '/' + dir_name)
        for dir_name in model_name:
            image_list.append(self.root1 + '/' + dir_name + '/' + self.dataset_name)
        print(image_list)
        return image_list

    def __getitem__(self, idx):
        image_idx = self.image_name[idx]
        result = []
        for dir_name in self.image_list:
            result.append(dir_name + '/' + image_idx)
        last_size = None
        for i in range(len(result)):
            suffix = ".png" if i > 0 else ".jpg"
            path = result[i] + suffix
            result[i] = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if last_size is None:
                last_size = result[i].shape
            if result[i].shape != last_size:
                result[i] = cv.resize(result[i], dsize=(last_size[1], last_size[0]))
            # print(last_size)
            # print(path, result[i].shape)

            # cv.imshow("", result[i])
            # cv.waitKey(0)
        images1 = np.hstack(result[0: 4])
        images2 = np.hstack(result[4: 8])
        images3 = np.hstack(result[8: 12])
        images4 = np.hstack(result[12: 16])
        images5 = np.hstack(result[16: 20])
        images = np.vstack([images1, images2, images3, images4, images5])
        window_name = str(image_idx)
        cv.namedWindow(window_name, 0)
        cv.resizeWindow(window_name, 1000, 1000)
        cv.imshow(window_name,images)
        cv.waitKey(0)
        cv.destroyAllWindows() 
        return idx
        
    def __len__(self):
        return self.image_name.__len__()
        
    def read(self, path=ROOT3):
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n').strip(" ")  
                self.result_name.append(line)
        print(self.result_name)
            
    def concat_result_name(self):
        rgb = self.root2 + '/' + "RGB"
        gt = self.root2 + '/' + "GT"
        depth = self.root2 + '/' + "depth"
        # models = [self.root2 + '/' + sub_dir for sub_dir in self.image_list]
        models = self.image_list
        # get all paths
        paths = []
        paths.extend(models)
        suffixes = [".jpg", ".png", ".bmp"]
        suffixes.extend([".png" for _ in range(17)])


        images = [] 
        for path, suffix in zip(paths, suffixes):
            row = []
            for sample_name in self.result_name:
                # images.append(cv.imread(path + '/' + sample_name + suffix, cv.))
                target_path: str = path + '/' + sample_name + suffix
                print(target_path)
                # os.system(fr"copy {target_path} {OUTPUT}\ ")
                if target_path.endswith(".jpg"):
                    # print(target_path)
                    img = cv.imread(target_path, cv.IMREAD_COLOR)
                    img = cv.resize(img, dsize=(224, 224))
                    row.append(img)
                else:
                    # print(target_path)
                    img = cv.imread(target_path, cv.IMREAD_GRAYSCALE)
                    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
                    img = cv.resize(img, dsize=(224, 224))
                    row.append(img)
            images.append(row)


        rows = []
        for ls in images:
            rows.append(np.hstack(ls))
        result = np.vstack(rows)
        cv.imwrite("./result.jpg", result)
        
    def write(self):
        pass

if __name__ == "__main__":
    image_list = ImageList()
    image_list.read()
    image_list.concat_result_name()
    # first = 30
    # size = image_list.__len__()
    # for i in range(first, size):
    #     image_list[i]
