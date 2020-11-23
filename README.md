# 人脸识别项目  Face_Recognition-new



##1、项目结构介绍
```
###config：OpenCV的分类器配置文件
###work_space/save:预训练模型
###data/Facebank 提供要检测的面部图像
```


##2、模块介绍
```
###take_pic.py:使用摄像头保存图片
###mtcnn.py:mtcnn 三级联网络结构
###model.py: 定义卷积，优化器，损失函数等
###demoForShow.py:人脸识别展示Demo模块，通过将模型的调用暴露成web服务，实现在浏览器端进行展示
###face_verify.py:调用摄像头来进行实时的人脸数据识别
###learner.py:定义训练、测试方法等
###Front_and_Display.py 连接oracle测试库，通过调用摄像头进行人脸识别，在HTTP中通过get请求返回对应人脸信息
###Front_and_Display_image.py 连接mysql测试库，通过在html中表单上传的方式，上传.jpg格式图片进行人脸识别，在HTTP中通过post请求返回对应人脸信息
###testMysql.py 连接mysql数据库，通过在html中表单上传的方式，上传.jpg格式图片进行人脸识别，在HTTP中通过post请求返回对应人脸信息
```


##3、项目依赖的第三方模块
```
###torch==0.4.0
numpy==1.14.5
matplotlib==2.1.2
tqdm==4.23.4
mxnet_cu90==1.2.1
scipy==1.0.0
bcolz==1.2.1
easydict==1.7
opencv_python==3.4.0.12
Pillow==5.2.0
mxnet==1.2.1.post1
scikit_learn==0.19.2
tensorboardX==1.2
torchvision==0.2.1
Flask
cx_Oracle
pymysql
```


##4、项目使用说明
```

##预训练模型下载  
链接：https://pan.baidu.com/s/1rCtHNJsZWaAF_H0UspdKWQ 
提取码：8csj 

下载成功后放入Face_Recognition-new\Face_Recognition\work_space\save目录下，程序即可正常运行


+  拍照可以，运行

  python take_pic.py -n name

  摄像头启动后，输入法记得调整成英文输入状态，然后点击一下q就会自动照一张相存储到data/Facebank 下面的指定文件夹里面，按T退出拍照
  如果相机中出现1个以上的人，则只会拍摄1张可能性最高的人脸
    
    
+ 或者可以将任何预先存在的照片放入facebank目录中，文件结构如下：

- facebank/
         name1/
             photo1.jpg
             photo2.jpg
             ...
         name2/
             photo1.jpg
             photo2.jpg
             ...
         .....
   如果目录中出现1张以上图片，则将计算平均嵌入
  
  
+开始测试

  python face_verify.py

就可以启动人脸识别项目的Demo了，这个模块会启动电脑的摄像头来实时地进行人脸识别。
- - -
  Front_and_Display_cors.py
  
连接fgos测试库，通过TD_VIP_TEST数据表获取所识别人脸信息

通过post请求，与前端进行交互，接收前端发送的二进制图片信息，转化为图片后进行人脸识别，以json格式将信息返回到前端
  ```

##5、html配置信息
 ```
Front_and_Display_image.py

在本地通过表单上传的方式进行人脸识别

新建一个文本文档，将以下配置信息放入即可
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
<form action="http://192.168.56.1:5000/ly" method="post" enctype="multipart/form-data">
    <input type="text" name="username">
    <input type="file" name="image">
    <input type="submit">
</form>
</body>
</html>
 ```
 
##6、项目API服务
  ```
 Front_and_Display.py 服务接口为http://IP:5000（其中，IP为自己的IP地址），通过调用摄像头进行人脸识别，返回mysql测试库中对应人脸信息
 
 Front_and_Display_image.py 通过在html中表单上传的方式，上传.jpg格式图片进行人脸识别，返回mysql测试库中对应人脸信息
 
 Front_and_Display_cors.py 实现前端与后端交互
 ```

