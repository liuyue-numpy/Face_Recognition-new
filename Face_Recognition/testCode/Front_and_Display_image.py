#!usr/bin/env python
#encoding:utf-8
from __future__ import division

import flask

'''
功能： 将人脸识别模型暴露为web服务接口，用于演示的demo
'''
import sys
sys.path.append("..")
import pymysql
from Model.config import get_config
from flask import Flask,jsonify, request
import argparse
from Model.mtcnn import MTCNN
from torch import optim
from Model.Learner import face_learner
from Model.utils import load_facebank, draw_box_name, prepare_facebank
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from PIL import Image
import io
import cx_Oracle


# REST 是Representational State Transfer的缩写，这是一种架构风格
# 调用flask初始化一个app
app=Flask(__name__)



@app.route("/")
def init():
    '''
    初始化接口
    http://IP:5000
    '''
    return u"人脸识别程序正常启动！"


def predict(image):
    # image = Image.open(img_path)
    # img=img_path#注意修改

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
    # image = Image.open(args.img_path)

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
            res = names[results[idx] + 1]
    except:
        print('detect')
        res = "unknown"


    return res





# 启动REST API,定义请求方式为 POST，表示向服务器传输数据
@app.route("/ly", methods=['POST'])
def get_frame():
    # Initialize the data dictionary that will be returned from the view.
    # 建立一个字典 data 来存储请求状态，初始化为 false
    # 记录请求的数据，并转换为字符串
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    # 判断是否是 POST 请求,记录请求使用的HTTP方法
    if flask.request.method == 'POST':
        # 判断是否能够得到从远端传过来的数据
        if flask.request.files.get("image"):
            # Read the image in PIL format
            # 得到从远方 POST 上来的数据
            image = flask.request.files["image"].read()
            # 将二进制的文件读取出来,再通过 PIL.Image.open 来读取这个图片，这样我们就解码了一张从远端传过来的图片了
            image = Image.open(io.BytesIO(image))
    # 记录请求中的表单数据
    userName = request.form['username']
    #image=request.files['image']
    #fileName='D:/image/'+image.filename
    #image.save(fileName)
    #print(image)
    print(userName)
    res=predict(image)
    if res == "unknown":
        data['predict'] = list()

        # Loop over the results and add them to the list of returned predictions
        # label_name = idx2label[label]
        # 将结果存到 data 中
        # r = {res}
        data['predict'].append(res)

        # Indicate that the request was a success.
        data["success"] = True
    else:

        """打开数据库"""
        #
        db = cx_Oracle.connect('fgos', 'bjrdfgos', '172.20.19.156:1521/fgosora')
        # 使用cursor()方法获取操作游标
        cursor = db.cursor()
        # sql = "SELECT * FROM names where name='%res'"
        # 使用execute方法执行SQL语句
        cursor.execute("SELECT * FROM TD_VIP_TEST where VIPU_VIP_NAME='%s'" % (res))

        # 使用 fetchone() 方法获取一条数据
        dat = cursor.fetchone()
        print(dat)
        # 关闭数据库连接
        db.close()
        data['predict'] = list()

        for row in dat:
            result = {}
            result['VIPU_OP_TM'] = dat[11]
            result['VIPU_TELE'] = dat[10]
            result['VIPU_LINE'] = dat[9]
            result['VIPU_GRADE'] = dat[8]
            result['VIPU_VIP_TITLE'] = dat[7]
            result['VIPU_PNR'] = dat[6]
            result['VIPU_DATE'] = dat[5]
            result['VIPU_TYPE_ARR_DEP'] = dat[4]
            result['VIPU_FLT_NO'] = dat[3]
            result['VIPU_ALCD_TW'] = dat[2]
            result['VIPU_ID'] = dat[1]
            result['VIPU_VIP_NAME'] = dat[0]




            #jsonData.append(result)
            #print
            # u'转换为列表字典的原始数据：', jsonData
        # Loop over the results and add them to the list of returned predictions
            # label_name = idx2label[label]
        # 将结果存到 data 中
        #r = {dat}
        data['predict'].append(result)

        # Indicate that the request was a success.
        data["success"] = True

    # Return the data dictionary as a JSON response.
    # 返回成 json 的文件
    return flask.jsonify(data)







if __name__ == "__main__":
    print('faceRegnitionDemo')
    #运行前需要将  0.0.0.0  替换为自己的IP地址
    app.run(host='192.168.56.1',port=5000)