
#调用一些相关的包
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
import numpy as np
import paddlehub as hub
matplotlib.use('TkAgg')

# S1  衣服图片抠图 ---------------------------------------------------------------------

# S3  获取关键点图像 ---------------------------------------------------------------------
module = hub.Module(name="human_pose_estimation_resnet50_mpii")
res = module.keypoint_detection(paths = ["./huanzhuang/human01.png"], visualization=True, output_dir='./huanzhuang/pic_output')

# res_img_path = './huanzhuang/pic_output/human01.jpg'
# img = mpimg.imread(res_img_path)
# plt.figure(figsize=(10, 10))
# plt.imshow(img)
# plt.axis('on')
# plt.show()
print(res)

# S4  换衣服 ---------------------------------------------------------------------
# 获取衣服位置 获取脖子以上位置
bottom_posx = res[0]["data"]["upper_neck"][0]
bottom_posy = res[0]["data"]["upper_neck"][1]
top_posx = res[0]["data"]["head_top"][0]
top_posy = res[0]["data"]["head_top"][1]
print(bottom_posx, bottom_posy)
print(top_posx, top_posy)

# 读取图片
Image1 = Image.open('./humanseg_output/1.png')
Image1copy = Image1.copy()

Image2 = Image.open('./huanzhuang/skirt.png')
Image2copy = Image2.copy()

# resize clothes
width, height = Image1copy.size
newsize = (int(width * 0.6), int(height * 0.9))
Image2copy = Image2.resize(newsize)

# 制定要粘贴左上角坐标
position = (int(bottom_posx * 0.5), int(bottom_posy * 0.6))  # ,right_posx, right_posy
print(position)
# 换衣服 ， 应该还有更好的方法进行照片合成
Image1copy.paste(Image2copy, position, Image2copy)  # 将翻转后图像region  粘贴到  原图im 中的 box位置

# 存为新文件
# Image1copy.save('./pic_output/newclothes.png')
Image1copy.save('./huanzhuang/pic_output/newclothes.png')

# 显示穿着新衣的照片
img = mpimg.imread('./huanzhuang/pic_output/newclothes.png')

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()