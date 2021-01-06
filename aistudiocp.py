import paddlehub as hub
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2


# 展示预测结果图片
test_img_path = "./image/test/test.jpg"
out_img_path = ""
img = mpimg.imread(test_img_path)
human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")
result = human_seg.segmentation(images=[mpimg.imread(test_img_path)],visualization=False,output_dir="./")
print(result)

# 将轮廓保存为图片
res_image = Image.fromarray(np.uint8(result[0]['data']))
res_image.save(f"newboy.jpg")