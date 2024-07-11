# 记得解压refer1，这是程序的数据集
# 因为舍友说手写数字比车牌识别简单就花了十分钟改造一下又写个手写数字的代码
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def plt_show1(image):
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    plt.imshow(image)
    plt.show()


def plt_show(image):
    plt.imshow(image, cmap='gray')
    plt.show()


# 第一步，先处理图片，对图片进行处理，处理过后根据轮廓判断来获取需要的车牌位置
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


relative_path = select_image_file_relative()
test_image = cv2.imread(relative_path)  # 导入图片
plt_show1(test_image)  # 打印图片


def get_car_card_img(image):
    '''
    创建一个函数对导入的图片进行操作，去获取图片中车牌的位置，得到位置后对图片进行处理获取车牌
    :param image: 传入一个图片
    :return: image：包含切割出的车牌图片
    '''
    image = process_img(image)  # 对图片进行处理

    # cv2.Sobel计算输入灰度图像 gray_image 的水平方向（x 方向）的 Sobel 边缘检测，
    # 并将结果存储在 Sobel_x 变量中。这个边缘检测结果图像是一个 16 位有符号整数图像，可能需要进一步处理才能用于显示或后续分析。
    Sobel_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    # Sobel_y = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1)

    # 通过 cv2.convertScaleAbs() 函数实现，该函数将 16 位整数图像转换为 8 位无符号整数图像，并将所有负值转换为正值（通过取绝对值）。
    absX = cv2.convertScaleAbs(Sobel_x)
    # absY = cv2.convertScaleAbs(Sobel_y)

    image = absX  # 更新原始图像变量为x方向的边缘信息图像
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)  # 使用Otsu阈值法对图像进行二值化处理

    # 创建一个矩形结构元素，用于形态学操作 ,这里的结构元素是17x5的矩形，常用于闭运算来填充小的间隙
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    # 对图像进行闭运算，填充小的间隙
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX, iterations=3)
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))  # 去除垂直方向的小白点
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))  # 去除水平方向的小白点

    image = cv2.dilate(image, kernelX)  # 膨胀
    image = cv2.erode(image, kernelX)  # 腐蚀
    image = cv2.erode(image, kernelY)  # 腐蚀
    image = cv2.dilate(image, kernelY)  # 膨胀
    image = cv2.medianBlur(image, 15)
    plt_show(image)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plt_show(image)
    # 绘制轮廓
    for item in contours:
        # cv2中，对一个轮廓（contour）调用cv2.boundingRect()函数时，它返回的是一个包含四个整数值的元组，
        # 这四个值分别代表边界矩形的左上角坐标（x, y）以及矩形的宽度（width）和高度（height）。
        rect = cv2.boundingRect(item)
        # 将元组中的值分别赋予给x, y, width, height
        x, y, width, height = rect
        # 根据得到的数值进行条件判断
        if (width > (height * 3)) and (width < (height * 4)):
            # 将图片的位置从原来的图片上进行截取
            image = test_image[y:y + height, x:x + width]
            return image


img = test_image.copy()
car_card_img = get_car_card_img(img)
plt_show1(car_card_img)


# 第二步，对截取到的车牌进行处理，将其分割得到每一个字符的图片，方便后续进行模版匹配。
def car_card_divide(image):
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
        # 筛选出高度大约是宽度1.8到3.5倍的轮廓
        if (word[3] > (word[2] * 1.7)) and (word[3] < (word[2] * 3.5)):
            # 根据边界矩形切割出对应的图像区域
            divide_image = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
            # 将切割出的图像区域添加到word_imges列表中
            word_imges.append(divide_image)
    return word_imges


img = car_card_img.copy()
word_img = car_card_divide(img)

for i, j in enumerate(word_img):
    plt.subplot(1, 9, i + 1)  # 生成八个子图
    plt.imshow(word_img[i], cmap='gray')  # 列表中取出索引为 i 的图像
plt.show()

template = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
            'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '藏', '川', '鄂', '甘', '赣', '贵', '桂', '黑', '沪',
            '吉', '冀', '津', '晋', '京', '辽', '鲁', '蒙', '闽', '宁', '青',
            '琼', '陕', '苏', '皖', '湘', '新', '渝', '豫', '粤', '云', '浙']


# 第三步，读取本地的文件创建模版进行匹配，根据匹配程度返回结果
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


def get_chinese_word_list():
    '''
    从指定的目录结构中读取一系列文件路径，并将它们作为列表返回。
    :return: chinese_list:包含文件中文字符串完整路径的列表
    '''
    chinese_list = []
    for i in range(34, 64):
        # 读取中文字符的目录下的所有文件路径
        c_word = read_directory("./refer1/" + template[i])
        chinese_list.append(c_word)  # 将读取到的文件路径列表添加到chinese_list中

    return chinese_list


def get_en_word_list():
    '''
    从指定的目录结构中读取一系列文件路径，并将它们作为列表返回。
    :return: english_list:包含文件英文字符串完整路径的列表
    '''
    english_list = []
    for i in range(10, 34):
        # 读取英文字符目录下的所有文件路径
        en_word = read_directory("./refer1/" + template[i])
        english_list.append(en_word)  # 将读取到的文件路径列表添加到english_list中
    return english_list


def get_en_num_word_list():
    word_list = []
    for i in range(0, 34):
        # 读取所有数字和英文字符目录下的所有文件路径
        word = read_directory("./refer1/" + template[i])
        word_list.append(word)  # 将读取到的文件路径列表添加到word_list中
    return word_list


# 通过上面的三个函数得到了三个模版
chinese_word_list = get_chinese_word_list()
en_word_list = get_en_word_list()
word_list1 = get_en_num_word_list()

print(chinese_word_list)


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
    results = []  # 用于存储每个图像的最佳匹配模板名称

    # 遍历图像列表
    for index, word_img in enumerate(word_imges):
        # 根据图像索引选择不同的模板列表和索引规则
        if index == 0:
            # 第一个图像使用中文模板列表
            word_list = chinese_word_list
            template_offset = 34  # 偏移量
        elif index == 1:
            # 第二个图像使用英文模板列表
            word_list = en_word_list
            template_offset = 10
        else:
            # 其他图像使用数字和英文的模板列表
            word_list = word_list1
            template_offset = 0

            # 初始化最佳得分列表
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
        r = template[template_offset + best_index]

        # 将最佳匹配的模板名称添加到结果列表中
        results.append(r)

    return results


word_imges = word_img.copy()
result = match_Template(word_imges)
print(f'识别出来的车牌为{result}')
