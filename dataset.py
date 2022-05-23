import os

import PIL.Image as Image
import albumentations as A
import numpy as np
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2

from config import opt

# 图像读入与预处理
transforms = A.Compose([
    A.Resize(opt.imagesize, opt.imagesize),
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ToTensorV2(),
])


class Animedataset(data.Dataset):
    def __init__(self):
        self.img_list = os.listdir(opt.data_path)

        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        img = os.path.join(opt.data_path,self.img_list[index])
        img = np.array(Image.open(img))
        img = self.transforms(image=img)["image"]
        return img
