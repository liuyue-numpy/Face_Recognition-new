import os
import cv2
from config import get_config
import time
import numpy as np
from flask import Flask,jsonify, request
import argparse
import torch
from mtcnn import MTCNN
from torch import optim
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image



#if __name__ == "__main__":
def predict( ):


    #img=img_path#注意修改

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save", action="store_true")
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54,
                                type=float)
    parser.add_argument("-u", "--update", default=True, help="whether perform update the facebank",
                                action="store_true")
    parser.add_argument("-tta", "--tta", default=False, help="whether testCode time augmentation",
                                action="store_true")
    parser.add_argument("-c", "--score", default=True, help="whether show the confidence score",
                                action="store_true")
    parser.add_argument('--img_path', '-p', default='1.jpg', type=str,
                        help='input the name of the recording person')
    args = parser.parse_args()
    mtcnn = MTCNN()
    conf = get_config(False)
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    image = Image.open('D:/code/Python/InsightFace-刘悦/InsightFace_Pytorch-master/1.jpg')

    if conf.device.type == 'cpu':
        learner.load_state(conf, 'ir_se50.pth', True, True)
    else:
        learner.load_state(conf, 'ir_se50.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    try:
        # image = Image.fromarray(img)
        # 利用mtcnn网络，对齐
        bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
        bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
        bboxes = bboxes.astype(int)
        bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
        results, score = learner.infer(conf, faces, targets, args.tta)
        print(results, score)
        for idx, bbox in enumerate(bboxes):
            print(names[results[idx] + 1])
            res="name:"+names[results[idx] + 1]
    except:
        print('detect')
        res="unknow"
    return res


if __name__ == "__main__":
    print(predict())
    #运行前需要将  0.0.0.0  替换为自己的IP地址

@app.route("/ly", methods=['POST'])
def get_frame():
    userName = request.form['username']
    image=request.files['image']
    fileName='D:/image/'+image.filename
    image.save(fileName)
    print(image)
    print(userName)
    res=predict(fileName)
    return res

