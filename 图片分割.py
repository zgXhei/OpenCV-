# 测试图片为tm1.png
# 写这个程序是为了完成题目，编写一个Jupyter Notebook使用OpenCV 找出样本图片中 题图 与 正文 的分界线并用红线标出。

import cv2
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog


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
height, width1 = test_image.shape[:2]
print(width1)
img = process_img(test_image)

Sobel_x = cv2.Sobel(img, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(Sobel_x)
img = absX
ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
plt_show(img)

contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for item in contours:
    # cv2中，对一个轮廓（contour）调用cv2.boundingRect()函数时，它返回的是一个包含四个整数值的元组，
    # 这四个值分别代表边界矩形的左上角坐标（x, y）以及矩形的宽度（width）和高度（height）。
    rect = cv2.boundingRect(item)
    # 将元组中的值分别赋予给x, y, width, height
    x, y, width, height = rect
    # 根据得到的数值进行条件判断
    if width > 900:
        print(y)
        image = test_image.copy()
        plt_show1(test_image)
        image1 = cv2.line(image, (0, y), (width1, y), (0, 0, 255), 10)
        plt_show1(image1)
