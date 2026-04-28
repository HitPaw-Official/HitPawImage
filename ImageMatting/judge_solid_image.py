from PIL import Image
import numpy as np
import cv2

# def is_solid_color(image, tolerance=10):
#     """
#     判断图片是否为纯色或接近纯色。

#     :param image: PIL.Image对象
#     :param tolerance: 颜色差异的容忍度，默认为10
#     :return: 如果图片是纯色或接近纯色，返回True；否则返回False
#     """
#     # 将图片转换为NumPy数组
#     img_array = np.array(image)
    
#     # 计算所有像素的平均颜色值
#     avg_color = np.mean(img_array, axis=(0, 1))
    
#     # 计算每个像素与平均颜色的差异
#     diff = np.abs(img_array - avg_color)
#     print(diff)
#     # 判断所有像素的颜色差异是否都在容忍度范围内
#     if np.all(diff <= tolerance):
#         return True
#     else:
#         return False
    
def is_solid_color(image, tolerance=20):
    """
    判断图片是否为纯色或接近纯色。

    :param image: PIL.Image对象
    :param tolerance: 颜色差异的容忍度，默认为10
    :return: 如果图片是纯色或接近纯色，返回True；否则返回False
    """ 
    # 计算所有像素的平均颜色值
    avg_color = np.mean(image)
    
    # 计算每个像素与平均颜色的差异
    diff = np.abs(image - avg_color)
    # 判断所有像素的颜色差异是否都在容忍度范围内
    if np.all(diff <= tolerance):
        return True
    else:
        return False


# 测试
if __name__ == "__main__":
    import os
    img_dir = r'F:\datasets\matting\test\test_zhanguang\zhanguang_test02_user_people'
    for name in os.listdir(img_dir):
        # image_path = r"G:\datasets\matting\test_bug\bug\hGf95T635A1OFXJWXddQBphF.jpg"  # 替换为你的图片路径
        image_path = os.path.join(img_dir, name)
        image = cv2.imread(image_path, 0)
        # image = Image.open(image_path)
        
        # 判断图片是否为纯色或接近纯色
        if is_solid_color(image):
            print("这张图片是纯色或接近纯色的: ", name)
        else:
            # print("这张图片不是纯色或接近纯色的。")
            continue