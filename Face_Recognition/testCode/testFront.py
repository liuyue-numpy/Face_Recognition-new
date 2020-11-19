import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy


# 解决cv2.putText绘制中文乱码
def cv2ImgAddText(frame, text, left, top, textColor=(0, 0, 255), textSize=20):
    if isinstance(frame, numpy.ndarray):  # 判断是否OpenCV图片类型
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(frame)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(frame), cv2.COLOR_RGB2BGR)

if __name__ == '__main__':
    img = cv2ImgAddText(cv2.imread('D:/code/Python/Face_Recognition-new/Face_Recognition/1.jpg'), "大家好，我是片天边的云彩", 10, 65, (0, 0 , 139), 20)
    cv2.imshow('show', img)
    if cv2.waitKey(100000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()