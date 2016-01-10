#!/usr/bin/env python3
# coding=utf-8

import cv2

# setup video capture

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")

def detectFaces(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)# 调用cv2的矩形函数，画矩形

#在原图像上画矩形，框出所有人脸。
#调用Image模块的draw方法，Image.open获取图像句柄，ImageDraw.Draw获取该图像的draw实例，然后调用该draw实例的rectangle方法画矩形(矩形的坐标即
#detectFaces返回的坐标)，outline是矩形线条颜色(B,G,R)。
#注：原始图像如果是灰度图，则去掉outline，因为灰度图没有RGB可言。drawEyes、detectSmiles也一样。
def drawFaces(im):
    faces = detectFaces(im)
    if faces:
        draw_rects(im, faces, (0,255,0)) # 画矩形标记
    return im


def main():

    while True:
        ret, im = cap.read()
        drawFaces(im)
        cv2.imshow('video test', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
        if key == ord(' '):
            cv2.imwrite('vid_result.jpg', im)

if __name__ == '__main__':
    main()
