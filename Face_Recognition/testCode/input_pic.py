import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe, Value, Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save", action="store_true")
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument("-u", "--update", default=False, help="whether perform update the facebank", action="store_true")
    parser.add_argument("-tta", "--tta",default=False,  help="whether testCode time augmentation", action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    parser.add_argument('--img_path', '-p', default='1.jpg', type=str, help='input the name of the recording person')
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
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

    image = Image.open(args.img_path)

    try:
        #                 image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        # image = Image.fromarray(frame)
        bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
        bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
        bboxes = bboxes.astype(int)
        bboxes = bboxes + [-1, -1, 1, 1]  # persoaaanal choice
        results, score = learner.infer(conf, faces, targets, args.tta)
        for idx, bbox in enumerate(bboxes):
            print('name:', names[results[idx] + 1])
            print('score:', score[idx])
            # if args.score:
            #     frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
            # else:
            #     frame = draw_box_name(bbox, names[results[idx] + 1], frame)
    except:
        print('detect error')
