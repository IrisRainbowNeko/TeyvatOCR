import os

import torch

os.environ["MODULE_PATH"] = "./"

import sys
sys.path.append('ABINet/')

import argparse
import easyocr
import cv2
import numpy as np
from ABINet.pred import Predictor
from tqdm import tqdm

parser = argparse.ArgumentParser(description='OF Generate')
parser.add_argument('-p', '--path', default='data_gen/imgs/kl.png', type=str)
parser.add_argument('-s', '--save_dir', default='output', type=str)
parser.add_argument('-ts', '--textsize', default=1.0, type=float)
parser.add_argument('-fk', '--fkeep', default=1, type=int)
parser.add_argument('-r', '--ratio_split', default=6.2, type=int)
parser.add_argument('--step', action='store_true')
args = parser.parse_args()

def cv_imread(file_path = ""):
    img_mat=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8), cv2.IMREAD_COLOR)
    return img_mat

def img_resize(image, size=(1920,1080)):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = size[0]
    height_new = size[1]
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new

def detect(reader, img_path):
    horizontal_list, free_list = reader.detect(img_path, text_threshold=0.5)
    boxes=horizontal_list[0]
    for rect in free_list[0]:
        x1,x2,y1,y2 = int(rect[0][0]), int(rect[2][0]), int(rect[0][1]), int(rect[2][1])
        if x2-x1>4 and y2-y1>4:
            boxes.append([x1,x2,y1,y2])
    return boxes

def proc_box(boxes, max_ratio=6.2):
    new_boxes=[]
    for box in boxes:
        x1, x2, y1, y2 = box
        dx,dy = x2-x1,y2-y1
        ratio = dx/dy
        if ratio<max_ratio:
            new_boxes.append(box)
        else:
            x_step=int(dy*max_ratio)
            for x in range(x1, x2-x_step, x_step):
                new_boxes.append([x,x+x_step,y1,y2])
            if (dx%x_step)/dy>0.3:
                new_boxes.append([x2-dx%x_step, x2, y1, y2])
    return new_boxes

def get_text_patch(img, boxes):
    #img=cv_imread(img_path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H,W,C = img.shape
    img_patchs=[img[max(0,box[2]):min(box[3]+1, H),max(0,box[0]):min(box[1]+1, W),:] for box in boxes]
    return img_patchs

def image_demo(file, size=(1920,1080)):
    img = cv_imread(file)
    img = img_resize(img, size)
    with torch.no_grad():
        boxes = detect(reader, img)
        boxes = proc_box(boxes, args.ratio_split)

        result = []
        img_patchs = get_text_patch(img, boxes)
        for patch, box in zip(img_patchs, boxes):
            text = predictor.pred(patch)
            result.append([box, text])

    image = img
    for item in result:
        box = item[0]
        x1, y1 = int(box[0]), int(box[2])
        x2, y2 = int(box[1]), int(box[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, item[1], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, args.textsize, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(file)), image)

def image_demo_step(file, size=(1920,1080), fkeep=1, fend=20):
    img = cv_imread(file)
    img = img_resize(img, size)
    H,W,C=img.shape

    with torch.no_grad():
        boxes = detect(reader, img)
        boxes = proc_box(boxes, args.ratio_split)

        result = []
        img_patchs = get_text_patch(img, boxes)
        for patch, box in zip(img_patchs, boxes):
            text = predictor.pred(patch)
            result.append([box, text])

    image = img
    vid_writer = cv2.VideoWriter(os.path.join(args.save_dir, os.path.basename(file)+'.mp4'), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (int(W), int(H)))

    def add_frame(box, text):
        frame=np.copy(image)
        x1, y1 = int(box[0]), int(box[2])
        x2, y2 = int(box[1]), int(box[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if len(text)>0:
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, args.textsize, (255, 0, 0), 2)
        for i in range(fkeep):
            vid_writer.write(frame)
        return frame

    for item in tqdm(result):
        box = item[0]
        text = item[1]
        for i in range(len(text)):
            add_frame(box, text[:i])
        frame = add_frame(box, text)
        image=frame

    for i in range(fend):
        vid_writer.write(image)

    vid_writer.release()

path = args.path
os.makedirs(args.save_dir, exist_ok=True)

reader = easyocr.Reader(['en'])
predictor=Predictor('ABINet/configs/train_abinet.yaml', 'ABINet/weights/train-abinet_8_9000.pth')

print('init ok')

def demo_one(file):
    if args.step:
        image_demo_step(file, fkeep=args.fkeep)
    else:
        image_demo(file)

if os.path.isfile(path):
    demo_one(path)
else:
    img_list=os.listdir(path)
    for img in img_list:
        print(img)
        demo_one(os.path.join(path,img))
print('ok')
#cv2.imshow('test', image)
#cv2.waitKey()