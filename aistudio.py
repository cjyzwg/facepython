import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
import cv2
import paddlehub as hub
from PIL import Image, ImageSequence, ImageDraw, ImageFont
from IPython.display import display, HTML
import numpy as np
import imageio
import os

# 测试图片路径和输出路径
base_path = '/Users/cj/Documents/code/python/'
test_path = base_path+'image/'
out_path = base_path+'image/output/'
# 待预测图片
test_img_path = ["1.jpg"]
test_img_path = [test_path + img for img in test_img_path]
img = mpimg.imread(test_img_path[0])
module = hub.Module(name="deeplabv3p_xception65_humanseg")
input_dict = {"image": test_img_path}
# execute predict and print the result
results = module.segmentation(data=input_dict)
for result in results:
    print(result)
# # 预测结果展示
out_img_path = base_path+'humanseg_output/' + os.path.basename(test_img_path[0]).split('.')[0] + '.png'
# print(out_img_path)
# img = mpimg.imread(out_img_path)
# plt.figure(figsize=(10, 10))
# plt.imshow(img)
# plt.axis('off')
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


output_path_img = out_path + 'blend_res_img.jpg'
blend_images('avatar/out.png', 'image/a.jpg', output_path_img)

# 展示合成图片
plt.figure(figsize=(10, 10))
img = mpimg.imread(output_path_img)
plt.imshow(img)
plt.axis('off')
plt.show()