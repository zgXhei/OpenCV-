import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


def plt_show(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def process_img(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)  # 高斯去噪
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)  # 灰度化处理
    return gray_image


# 定义一个函数，执行时弹出界面选择图片进行识别
def select_image_file_relative():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp"), ("All Files", "*.*")])
    if file_path:
        # 获取当前工作目录的绝对路径
        cwd = os.getcwd()
        # 计算相对于当前工作目录的相对路径
        relative_path = os.path.relpath(file_path, cwd)
        return relative_path

    # 调用函数并打印相对路径


# 第一步，对得到的数字进行处理，将其分割得到每一个字符的图片，方便后续进行模版匹配。
def num_divide(image):
    '''
    创建一个方法对车牌进行分割，对每一个字符进行提取存入列表方便后续进行模板匹配
    :param image: 传入车牌图片
    :return: word_imges:包含分割完后字符图片的列表
    '''
    image = process_img(image)
    plt_show(image)
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    plt_show(image)
    # 计算图像和黑白点个数,对牌照进行一个颜色的转换,让号码变为白色方便后续的模型训练
    area_white = 0
    area_black = 0
    height, width = image.shape
    print(image.shape)
    for i in range(height):
        for j in range(width):
            if image[i, j] == 255:
                area_white += 1
            else:
                area_black += 1
    # 如果白色像素的面积大于黑色像素的面积
    if area_white > area_black:
        # 使用OTSU或者THRESH_BINARY_INV阈值方法自动计算阈值，并将图像二值化（逆二值化）
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        plt_show(image)

    # 定义一个2x2的矩形结构元素用于膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 对图像进行膨胀操作
    image = cv2.dilate(image, kernel)

    plt_show(image)
    # 查找图像中的外部轮廓
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 在原始图像上绘制找到的轮廓
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    plt_show(image)
    # 初始化列表来保存轮廓的边界矩形信息和对应的图像
    words = []
    word_imges = []

    # 遍历轮廓
    for item in contours:
        word = []
        rect = cv2.boundingRect(item)
        x, y, width, height = rect
        word.append(x)
        word.append(y)
        word.append(width)
        word.append(height)
        words.append(word)

    # 根据边界矩形的x坐标进行排序（保持从左到右的顺序）
    words = sorted(words, key=lambda s: s[0], reverse=False)
    print(words)

    # 遍历排序后的边界矩形信息
    for word in words:
        # 根据边界矩形切割出对应的图像区域
        divide_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
        # 将切割出的图像区域添加到word_imges列表中
        word_imges.append(divide_image)
    return word_imges


template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 第二步，建立模版，进行模版匹配
def read_directory(directory_name):
    '''
    读取指定目录中的所有文件，并返回包含这些文件完整路径的列表。
    :param directory_name:要读取的目录名称。
    :return:img_list: 包含目录中所有文件完整路径的列表。
    '''
    img_list = []  # 初始化一个空列表，用于存储文件的完整路径
    for filename in os.listdir(directory_name):
        # 构造文件的完整路径，并将它添加到img_list列表中
        img_list.append(directory_name + "/" + filename)
    return img_list


def get_num_word_list():
    word_list = []
    for i in range(0, 9):
        # 读取所有数字和英文字符目录下的所有文件路径
        word = read_directory("./refer1/" + template[i])
        word_list.append(word)  # 将读取到的文件路径列表添加到word_list中
    return word_list


word_list1 = get_num_word_list()


def template_score(template, image):
    '''
    计算模板图像与给定图像之间的匹配度得分。
    :param template:模板图像
    :param image:要与模板图像匹配的图像
    :return:result[0][0]:模板匹配得分
    '''
    # 从文件中读取模板图像，并将其解码为 NumPy 数组 (尝试过不解码，会产生报错)
    template_img = cv2.imdecode(np.fromfile(template, dtype=np.uint8), 1)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)  # 将原始图像转换为灰度图像
    # 使用 Otsu's 方法进行阈值处理，将图像二值化
    ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
    image_ = image.copy()  # 复制输入图像（避免在原始图像上进行操作）
    # 获取输入图像的高度和宽度
    height, width = image_.shape
    # 将模板图像调整为与输入图像相同的大小
    template_img = cv2.resize(template_img, (width, height))
    # 使用模板匹配函数计算匹配度
    result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF)
    return result[0][0]


def match_Template(word_imges):
    """
    对图像列表中的每个图像进行模板匹配，并返回最佳匹配的模板名称列表。
    参数:word_imges (list): 包含多个图像的列表。
    返回:list: 包含每个图像最佳匹配模板名称的列表。
    """
    results = []
    for index, word_img in enumerate(word_imges):
        # 根据图像索引选择不同的模板列表和索引规则
        word_list = word_list1
        best_scores = []

        # 遍历模板列表中的每个模板组
        for template_group in word_list:
            # 初始化当前模板组的得分列表
            scores = []

            # 遍历当前模板组中的每个模板
            for template_path in template_group:
                # 计算模板与当前图像的匹配得分
                score = template_score(template_path, word_img)
                scores.append(score)

                # 找到当前模板组的最高得分，并添加到最佳得分列表中
            best_scores.append(max(scores))

            # 找到所有模板组中的最高得分，并获取其索引
        best_index = best_scores.index(max(best_scores))

        # 根据索引和偏移量从template列表中获取最佳匹配的模板名称
        r = template[best_index]

        # 将最佳匹配的模板名称添加到结果列表中
        results.append(r)

    return results


relative_path = select_image_file_relative()
test_image = cv2.imread(relative_path)  # 导入图片
image = test_image.copy()

img = image.copy()
word_img = num_divide(img)

for i, j in enumerate(word_img):
    plt.subplot(1, 11, i + 1)  # 生成11个子图，刚好是电话号码
    plt.imshow(word_img[i], cmap='gray')  # 列表中取出索引为 i 的图像
plt.show()
result = match_Template(word_img)
print(f'识别出来的数字为{result}')
# 为了证明手写数字识别比车牌识别简单所以又写一个手写数字识别，实现过程和车牌识别差不多，但车牌识别就是比手写数字难
