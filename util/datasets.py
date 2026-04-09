# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
import glob
import numpy as np
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset,DataLoader  #用于定义数据读取器
import pickle

from PIL import Image


from PIL import ImageFile


#构造函数指明，需要传入图片和标签，同时可以定义数据增强
class MyCustomData(Dataset):
    def __init__(self,is_train, args):  #x输入的是[路径]，y是对应的[标签]
        root = os.path.join(args.data_path, is_train, 'dataset.pkl')
        with open(root, 'rb') as file:
            dataset_dic = pickle.load(file)
        path1 = dataset_dic['1']
        path0 = dataset_dic['0']
        label1 = ['1']*len(path1)
        label0 = ['0']*len(path0)
        # 训练集欠采样：随机抽取与阳性等量的阴性，实现 1:1 平衡（固定 seed 保证分布式时各进程一致）
        if is_train == 'train' and getattr(args, 'balance_train', False):
            n_pos = len(path1)
            if len(path0) > n_pos:
                rng = np.random.default_rng(getattr(args, 'seed', 0))
                idx = rng.choice(len(path0), size=n_pos, replace=False)
                path0 = [path0[i] for i in idx]
                label0 = ['0']*n_pos
                print(f'[balance_train] 阴性欠采样: {len(path0)} (与阳性 {n_pos} 等量, 1:1)')
        label1.extend(label0)
        self.y = label1
        path1.extend(path0)
        self.x = path1
        self.is_train = is_train
        self.args = args
        self.transform = self.build_transform()
        
    def __getitem__(self,index): #把图片读成矩阵的形式，标签转成数字的形式
        data=self.x[index]  #一张图片的路径
        label=self.y[index] #一张图片的标签
        image=Image.open(data).convert("RGB")
        #print(image)
        label_temp=0
        if label=="0":label_temp=0
        elif label=="1":label_temp=1
        if self.transform is not None:
            image=self.transform(image)
        return data,image,label_temp
    def __len__(self):
        return len(self.x)


    # def build_transform(self):
    #     mean = IMAGENET_DEFAULT_MEAN
    #     std = IMAGENET_DEFAULT_STD
    #     # train transform
    #     if self.is_train=='train':
    #         # this should always dispatch to transforms_imagenet_train
    #         transform = create_transform(
    #             input_size=self.args.input_size,
    #             is_training=True,
    #             color_jitter=self.args.color_jitter,
    #             auto_augment=self.args.aa,
    #             interpolation='bicubic',
    #             re_prob=self.args.reprob,
    #             re_mode=self.args.remode,
    #             re_count=self.args.recount,
    #             mean=mean,
    #             std=std,
    #         )
    #         return transform

    #     # eval transform
    #     t = []
    #     if self.args.input_size <= 224:
    #         crop_pct = 224 / 256
    #     else:
    #         crop_pct = 1.0
    #     size = int(self.args.input_size / crop_pct)
    #     t.append(
    #         transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    #     )
    #     t.append(transforms.CenterCrop(self.args.input_size))
    #     t.append(transforms.ToTensor())
    #     t.append(transforms.Normalize(mean, std))
    #     return transforms.Compose(t)
    

    def build_transform(self):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        if self.is_train=='train':
            transform = transforms.Compose([
                transforms.Resize(224),  
                transforms.CenterCrop(224),       
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=(-180, 180)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(),
                # transforms.RandomApply([
                #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) 
                #     ], p=0.8), 
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0))  # not strengthened
                ], p=0.2), 
                # transforms.RandomApply([
                #         transforms.RandomResizedCrop(args.input_size, scale=(0.64, 1.0), ratio=(3/4, 4/3))
                #     ], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.09), ratio=(0.3, 3.3), value=0, inplace=False)
                ])
        else:
            transform = transforms.Compose([
                # CropCenterSquare(),
                transforms.Resize(224),
                transforms.CenterCrop(224), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),                
                ])

        return transform