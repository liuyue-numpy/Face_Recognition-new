import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import sys
sys.path.append("..")
from mtcnn_pytorch.src.get_nets import PNet, RNet, ONet
from mtcnn_pytorch.src.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from mtcnn_pytorch.src.first_stage import run_first_stage
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
device = torch.device("cpu")
# device = 'cpu'

import math



class MTCNN():
    def __init__(self):
        self.pnet = PNet().to(device)
        self.rnet = RNet().to(device)
        self.onet = ONet().to(device)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.refrence = get_reference_facial_points(default_square= True)
        
    def align(self, img):
        _, landmarks = self.detect_faces(img)
        facial5points = [[landmarks[0][j],landmarks[0][j+5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
        return Image.fromarray(warped_face)
    
    def align_multi(self, img, limit=None, min_face_size=30.0):
        # 框和人脸关键点定位
        boxes, landmarks = self.detect_faces(img, min_face_size)
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[j],landmark[j+5]] for j in range(5)]
            # 人脸平均，得到平均脸
            warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=(112,112))
            faces.append(Image.fromarray(warped_face))
        return boxes, faces

    def detect_faces(self, image, min_face_size=20.0,
                     thresholds=[0.6, 0.7, 0.8],
                     nms_thresholds=[0.7, 0.7, 0.7]):
        """
        Arguments:
            image: an instance of PIL.Image.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size/min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m*factor**factor_count)
            min_length *= factor
            factor_count += 1


        # STAGE 1：人脸分类，用人脸图像和非人脸图像进行训练，先使用卷积网络获取人脸框和边界框回归向量(Bounding Box Regression)，
        # 再用边界框回归向量进行人脸框校准，之后用非极大抑制(NMS)去除重合度高的候选框
        #   将原图压缩到不同尺度的图像金字塔输入到P-Net中(可以更大程度确保不同大小的人脸都被检测到)。
    # 取图片12x12x3的区域作为输入，通过卷积和池化输出：人脸分类(2个输出)、边界框回归(回归框的左上和右下坐标，4个输出)、人脸关键点定位(P-Net中不输出)。
        # it will be returned
        bounding_boxes = []

        with torch.no_grad():
            # run P-Net on different scales
            for s in scales:
                boxes = run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0])
                #if boxes is not None:
                 #   print("boxes.shape:", boxes.shape)
                bounding_boxes.append(boxes)

            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            bounding_boxes = np.vstack(bounding_boxes)

            # print('number of bounding boxes:', bounding_boxes.shape, len(bounding_boxes))

            keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])

            bounding_boxes = bounding_boxes[keep]

            # use offsets predicted by pnet to transform bounding boxes
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            # shape [n_boxes, 5]

            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 2：边框回归，将P-Net得到的回归框输入到R-Net中，再对回归框进行校准，并使用NMS去重
            #    P-Net网络简单速度快，但准确率较低，所有用R-Net来进一步做检测。把P-Net人脸检测窗口resize为24x24x3大小，再传入R-Net中，可消除很多误判。
    # R-Net的输出和P-Net的输出结果一样：人脸分类(2个输出)、边界框回归(回归框的左上和右下坐标，4个输出)、人脸关键点定位(R-Net中不输出)。

            img_boxes = get_image_boxes(bounding_boxes, image, size=24)
            img_boxes = torch.FloatTensor(img_boxes).to(device)

            output = self.rnet(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]

            keep = nms(bounding_boxes, nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 3：人脸关键点定位，    把R-Net得到窗口resize为48x48x3大小，再传入O-Net进行检测。
            #     O-Net的输出为：人脸分类(2个输出)、边界框回归(回归框的左上和右下坐标，4个输出)、人脸关键点定位(10个输出)。

            img_boxes = get_image_boxes(bounding_boxes, image, size=48)
            if len(img_boxes) == 0:
                return [], []
            img_boxes = torch.FloatTensor(img_boxes).to(device)
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]

        return bounding_boxes, landmarks
