import cv2
import paddlehub as hub
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
matplotlib.use('TkAgg')

output_path = './humanseg_output/'
output_img_name = '1.png'
output_img = output_path+output_img_name
module = hub.Module(name="deeplabv3p_xception65_humanseg")

# execute predict and print the result
results = module.segmentation(images=[cv2.imread('./image/human01.png')],visualization=True,output_dir=output_path)
for result in results:
    os.rename(result['save_path'], output_img)
    # print(result)
    # print(result['save_path'])

# 扣取人像
# img = mpimg.imread(output_img)
# plt.figure(figsize=(10,10))
# plt.imshow(img)
# plt.axis('on')
# plt.show()


# 合成函数
def blend_images(fore_image, base_image, output_path):
    """
    将抠出的人物图像换背景
    fore_image: 前景图片，抠出的人物图片
    base_image: 背景图片
    """
    # 读入图片
    base_image = Image.open(base_image).convert('RGB')
    fore_image = Image.open(fore_image).resize(base_image.size)

    # 图片加权合成
    scope_map = np.array(fore_image)[:, :, -1] / 255
    scope_map = scope_map[:, :, np.newaxis]
    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    res_image = np.multiply(scope_map, np.array(fore_image)[:, :, :3]) + np.multiply((1 - scope_map),
                                                                                     np.array(base_image))

    # 保存图片
    res_image = Image.fromarray(np.uint8(res_image))
    res_image.save(output_path)
    return res_image


output_path_img = output_path + 'blend_res_img.jpg'
blend_images(output_img, './image/a.jpg', output_path_img)

# 展示合成图片
plt.figure(figsize=(10, 10))
img = mpimg.imread(output_path_img)
plt.imshow(img)
plt.axis('off')
plt.show()