import Augmentor
import os
import glob
from PIL import Image, ImageEnhance
import numpy as np  # 导入numpy库以生成随机数

# 定义输入图像和掩码目录
image_dir = "data/neuseg/train/images"
mask_dir = "data/neuseg/train/mask"

# 创建输出目录
output_image_dir = "data/neuseg/train/augmented_images"
output_mask_dir = "data/neuseg/train/augmented_masks"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# 获取所有图像和掩码文件
image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
mask_files = glob.glob(os.path.join(mask_dir, "*.png"))

# 检查图像和掩码的数量是否匹配
if len(image_files) != len(mask_files):
    raise ValueError("Number of images and masks do not match!")

# 添加自定义变换
def augment_image_and_mask(image, mask):
    # 旋转（最大左旋10°，最大右旋10°，80%的概率）
    if np.random.rand() < 0.8:  # 使用numpy的随机数生成
        angle = np.random.uniform(-10, 10)
        image = image.rotate(angle)
        mask = mask.rotate(angle)

    # 垂直翻转（50%的概率）
    if np.random.rand() < 1:
        image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(method=Image.FLIP_LEFT_RIGHT)

    # 随机对比度和亮度增强（30%的概率）
    # if np.random.rand() < 0.3:
    #     factor = np.random.uniform(1.0, 1.5)
    #     image = ImageEnhance.Contrast(image).enhance(factor)

    # if np.random.rand() < 0.3:
    #     factor = np.random.uniform(0.5, 1.5)
    #     image = ImageEnhance.Brightness(image).enhance(factor)

    # 随机放大和裁切（30%的概率）
    # if np.random.rand() < 0.3:
    #     image = image.resize((int(image.size[0] * 1.2), int(image.size[1] * 1.2)))
    #     mask = mask.resize((int(mask.size[0] * 1.2), int(mask.size[1] * 1.2)))

    # 裁切85%
    # if np.random.rand() < 0.3:
    #     image = image.crop((0, 0, int(image.size[0] * 0.85), int(image.size[1] * 0.85)))
    #     mask = mask.crop((0, 0, int(mask.size[0] * 0.85), int(mask.size[1] * 0.85)))

    # 改变图像大小（调整为256x256）
    # image = image.resize((256, 256))
    # mask = mask.resize((256, 256))

    return image, mask

# 逐一处理图像和掩码
for img_file, mask_file in zip(image_files, mask_files):
    img = Image.open(img_file)
    mask = Image.open(mask_file)

    augmented_img, augmented_mask = augment_image_and_mask(img, mask)

    # 为增强后的图像和掩码生成新的文件名
    new_img_name = os.path.splitext(os.path.basename(img_file))[0] + "_new.jpg"
    new_mask_name = os.path.splitext(os.path.basename(mask_file))[0] + "_new.png"

    # 保存增强后的图像和掩码
    augmented_img.save(os.path.join(output_image_dir, new_img_name))
    augmented_mask.save(os.path.join(output_mask_dir, new_mask_name))

print("Data augmentation completed for images and masks!")