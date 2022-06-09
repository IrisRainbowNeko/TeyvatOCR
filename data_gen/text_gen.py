import time

from trdg.generators import GeneratorFromRandom, GeneratorFromDict
from trdg.generators.from_dict import create_strings_from_dict
from trdg.generators.from_random import create_strings_randomly
import os
import csv
import cv2
import random
from tqdm import tqdm
import itertools
import numpy as np
import albumentations as A
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

trans=A.Compose([
    A.OneOf([
        A.IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
        A.GaussNoise(),    # 将高斯噪声应用于输入图像。
    ], p=0.5),
    A.OneOf([
            # 模糊相关操作
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
    A.OneOf([
        # 锐化、浮雕等操作
        A.CLAHE(clip_limit=2),
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.3),
    A.RGBShift(p=0.2),
    A.RandomBrightness(p=0.2),
    A.ChannelShuffle(p=0.2)
])

class GeneratorFromDictA(GeneratorFromRandom):
    def __init__(self, *args, **kwargs):
        super().__init__(use_numbers=False, use_symbols=False, margins=(0, 0, 0, 0), *args, **kwargs)
        self.generator.strings = [self.keep_alpha(x) for x in self.generator.strings]

    def keep_alpha(self, s):
        words=[''.join(filter(str.isalpha, x)) for x in s.split()]

        return '0'.join(words).upper()

    def __iter__(self):
        return self

    def __len__(self):
        return self.count

    def next(self):
        if self.generator.generated_count >= 999:
            self.generator.strings = [self.keep_alpha(x) for x in create_strings_randomly(
                self.length,
                self.allow_variable,
                1000,
                self.use_letters,
                self.use_numbers,
                self.use_symbols,
                self.language,
            )]
            #self.generator.strings = [self.keep_alpha(x) for x in create_strings_from_dict(
            #    self.length, self.allow_variable, 1000, self.dict
            #)]
        self.generator.distorsion_type = np.random.choice(3, 1, p=[0.7, 0.15, 0.15])
        self.generator.character_spacing = random.randint(0,6)
        return self.generator.next()


dir='../data/val'
#dir='../data/genshin'
n_th=8
n_imgs=5000

generator_list = [GeneratorFromDictA(
    fonts=['./font/GenShin.otf'],
    count=n_imgs//n_th,
    length=1,
    size=64,
    skewing_angle=5,
    random_skew=True,
    blur=1,
    random_blur=True,
    distorsion_type=1,
    background_type=3,
    text_color='#000000,#FFFFFF',
    image_dir='./images'
) for i in range(n_th)]

data_dict={}
pbar = tqdm(total=n_imgs)

def gen(idx, generator):
    img, lbl = generator.next()
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = trans(image=img)['image']
    # img.save(os.path.join(dir, f'{idx}.jpg'))
    cv2.imwrite(os.path.join(dir, f'{idx}.jpg'), img)
    #data.append([f'{idx}.jpg', lbl.replace('0', ' ')])
    data_dict[f'{idx}.jpg']=lbl.replace('0', ' ')
    pbar.update(1)


pool = ThreadPoolExecutor(max_workers=n_th)
all_task = [pool.submit(gen, idx, generator_list[idx%n_th]) for idx in range(n_imgs)]
wait(all_task, timeout=None, return_when=ALL_COMPLETED)

time.sleep(10)
with open(os.path.join(dir, f'labels.txt'),'w', newline='')as f:
    for k,v in data_dict.items():
        f.write(f'{k}\t{v}\n')