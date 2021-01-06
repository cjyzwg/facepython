import cv2
import paddlehub as hub
import numpy as np
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')



class face_Seg(object):
    '''
    imgFile：原始图片数据
    origLandMark：第一次detection的坐标
    faceImg：裁剪后的图片
    resLandMark：修改后的坐标
    mapImg：分割后的标注图片
    resFaceImg：最终分割后图片
    '''

    def __init__(self, imgFile):
        self.imgFile = imgFile
        self.origLandMark = None
        self.faceImg = None
        self.resLandMark = None
        self.mapImg = None
        self.resFaceImg = None
        self.oriMapImg = None
        self.orgBox = None

    @staticmethod
    def keypoint_detection(images):
        return hub.Module(name="face_landmark_localization").keypoint_detection(images=images)

    @staticmethod
    def human_parser(images):
        return hub.Module(name="ace2p").segmentation(images=images)

    '''
    获取原始图像
    '''

    def getOrgImgFile(self):
        return self.imgFile

    '''
    获取面部监测点信息
    '''

    def getOrLandMark(self):
        if np.all(self.origLandMark == None):
            try:
                res = self.keypoint_detection(images=[self.imgFile])
            except:
                print("无法打开图片或检测有问题！")
                return None
            self.origLandMark = np.array(res[0]['data'][0])
        return self.origLandMark

    '''
    获取面部图片的裁剪后图像
    '''

    def getFaceImg(self):
        if np.all(self.faceImg == None):
            LandMark = self.getOrLandMark()
            x = LandMark[:, 0]
            y = LandMark[:, 1]
            self.orgBox = (int(y.min()), int(y.max()), int(x.min()), int(x.max()))
            self.faceImg = self.imgFile[int(y.min()):int(y.max()), int(x.min()):int(x.max())]
            resLM = list(map(lambda a: [a[0] - int(x.min()), a[1] - int(y.min())], LandMark))
            self.resLandMark = np.array(resLM)
        return self.faceImg

    '''
    获取原图中截取的部分
    '''

    def getOrgBox(self):
        if self.orgBox == None:
            self.getFaceImg()
        return self.orgBox

    '''
    获取分割图片标记
    '''

    def getMapImg(self):
        if np.all(self.mapImg == None):
            res = self.human_parser(images=[self.getFaceImg()])
            self.mapImg = np.array(res[0]['data'])
        return self.mapImg

    '''
    仅返回脸部区域分割结果，和对应脸部区域的特征点坐标
    '''

    def getResult(self, savepath=None):
        if np.all(self.resFaceImg == None):
            mapimg = self.getMapImg()
            self.resFaceImg = self.getFaceImg().copy()
            X, Y = mapimg.shape
            for i in range(X):
                for j in range(Y):
                    if mapimg[i, j] != 13:
                        self.resFaceImg[i, j] = [255, 255, 255]
        if savepath != None:
            cv2.imwrite(savepath, self.resFaceImg)
            print("成功保存图片！")
        return self.resFaceImg, self.resLandMark

    '''
    将分割图片扩展到原始图片大小，并返回
    '''

    def getOriMapImg(self):
        if np.all(self.oriMapImg == None):
            X, Y, Z = self.imgFile.shape
            mapimg = self.getMapImg()
            self.oriMapImg = np.zeros((X, Y))
            LandMark = self.getOrLandMark()
            x = LandMark[:, 0]
            y = LandMark[:, 1]
            for i in range(X):
                for j in range(Y):
                    if x.min() < i < x.max() and y.min() < j < y.max():
                        self.oriMapImg[j, i] = mapimg[j - int(y.min()) - 1, i - int(x.min()) - 1]
        return self.oriMapImg


'''
计算两个向量的夹角，用于校准两张图片的脸部轴心位置
'''


def angle_between(v1, v2):
    angle1 = math.atan2(v1[0], v1[1])
    angle2 = math.atan2(v2[0], v2[1])
    return (angle2 - angle1) / math.pi * 180


'''
旋转图片并对图片矫正
'''


def rotate_bound(image, angle, center=None):
    (h, w) = image.shape[:2]
    # 定义旋转中心，可以选择自定义的中心进行旋转
    if np.all(center == None):
        (cX, cY) = (w // 2, h // 2)
    else:
        (cX, cY) = (int(center[0]), int(center[1]))

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


# 面部基本点检测，男神的脸
tar = face_Seg.keypoint_detection(images = [cv2.imread('./tar.jpg')])
LandMark = np.array(tar[0]['data'][0])
# 面部基本点检测，背景图片
pic = face_Seg(cv2.imread('./pic.png'))
baseLandMark = pic.getOrLandMark()
# 获取两个面部轴心向量
lineA = LandMark[27] - LandMark[8]
lineB = baseLandMark[27] -baseLandMark[8]
angle = angle_between(lineB,lineA)
# 旋转男神图像使得图像与背景图像的面部轴心夹角一致
tar = rotate_bound(cv2.imread('./tar.jpg'), angle,LandMark[8])
# 将旋转后图像放入face_Seg类中方便进一步处理
tar = face_Seg(tar)
# 获取面部切割结果并将面部切割的结果保存为face.jpg
FaceImg, LandMark = tar.getResult('face.jpg')
# 原图片处理,与上面一样
picFaceImg, picLandMark = pic.getResult('picface.jpg')
# 将两个脸的大小调整到一致
X,Y,Z = picFaceImg.shape
FaceImg = cv2.resize(FaceImg, (X,Y))
# 图片合并过程
result = pic.getOrgImgFile()
mask = 255 * np.ones(FaceImg.shape, FaceImg.dtype)
center = pic.getOrgBox()
center = (int((center[2]+center[3])/2) ,int((center[0]+center[1])/2))
flags = cv2.NORMAL_CLONE
FaceImg1 = FaceImg+5 # 调整亮度，是图像与背景融合更好
output = cv2.seamlessClone(FaceImg1, result, mask, center, flags)
plt.imshow(output)
cv2.imwrite('OK.jpg',output)