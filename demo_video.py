import os

import torch

os.environ["MODULE_PATH"] = "./"

import sys
sys.path.append('ABINet/')

import argparse
import easyocr
import cv2
import time
import numpy as np
from ABINet.pred import Predictor
from tqdm import tqdm

parser = argparse.ArgumentParser(description='OF Generate')
parser.add_argument('-p', '--path', default='video/tvt2.mp4', type=str)
parser.add_argument('-ts', '--textsize', default=1.0, type=float)
parser.add_argument('-s', '--skip', default=5, type=int)
parser.add_argument('-r', '--ratio_split', default=6.2, type=int)
args = parser.parse_args()

def cv_imread(file_path = ""):
    img_mat=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
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
            new_boxes.append([x2-dx%x_step, x2, y1, y2])
    return new_boxes

def get_text_patch(img, boxes):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H,W,C = img.shape
    img_patchs=[img[max(0,box[2]):min(box[3]+1, H),max(0,box[0]):min(box[1]+1, W),:] for box in boxes]
    return img_patchs

def video_demo(reader, predictor, save_folder, path, skip=5, size=(1920,1080)):
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, path.split("/")[-1])

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(size[0]), int(size[1]))
    )
    count=0
    pbar = tqdm(total=frame_count)
    result=None
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            frame = img_resize(frame, size)
            if count%skip==0:
                boxes = detect(reader, frame)
                boxes = proc_box(boxes, args.ratio_split)

                result = []
                img_patchs = get_text_patch(frame, boxes)
                for patch, box in zip(img_patchs, boxes):
                    text = predictor.pred(patch)
                    result.append([box, text])

            result_frame=frame
            for item in result:
                box = item[0]
                x1, y1 = int(box[0]), int(box[2])
                x2, y2 = int(box[1]), int(box[3])
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_frame, item[1], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, args.textsize, (255, 0, 0), 2)

            vid_writer.write(result_frame)
            ch = cv2.waitKey(1)

            #print(f'{count}/{frame_count}')
            pbar.update(1)
            count += 1
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

reader = easyocr.Reader(['en'])
predictor=Predictor('ABINet/configs/train_abinet.yaml', 'ABINet/weights/train-abinet_8_9000.pth')

print('init ok')

with torch.no_grad():
    video_demo(reader, predictor, 'output', args.path, skip=args.skip)
