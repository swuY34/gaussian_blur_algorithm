from PIL import Image
from scipy import signal
import numpy as np
import math
import matplotlib.pyplot as plt


def create_gaussian_filter(kernel_rad: int):
    kernel = np.zeros((kernel_rad * 2 + 1, kernel_rad * 2 + 1))  # 创建卷积核
    sigma = 0.5 * kernel_rad

    for x in range(-kernel_rad, kernel_rad + 1):
        for y in range(-kernel_rad, kernel_rad + 1):
            kernel[x + kernel_rad, y + kernel_rad] = (1 / (2 * math.pi * sigma ** 2)) * math.exp(
                -((x ** 2 + y ** 2) / (2 * sigma ** 2)))

    return kernel / np.sum(kernel)


def image_to_gaussian(img):
    gaussian_filter = create_gaussian_filter(10)  # kernel_rad = 3

    print(gaussian_filter)

    resR = signal.convolve2d(img[:, :, 0], gaussian_filter, mode='valid')
    resG = signal.convolve2d(img[:, :, 1], gaussian_filter, mode='valid')
    resB = signal.convolve2d(img[:, :, 2], gaussian_filter, mode='valid')

    result_image = np.zeros((resR.shape[0], resR.shape[1], 3))
    result_image[:, :, 0] = resR[:, :]
    result_image[:, :, 1] = resG[:, :]
    result_image[:, :, 2] = resB[:, :]

    return result_image


def main():
    # 首先要做的就是将图片读取成 arrayList
    image_array = np.asarray(Image.open("original_image.png")) / 255

    blur_image = image_to_gaussian(image_array)
    plt.imshow(blur_image)
    plt.show()
    plt.imsave("0001.png", blur_image)


if __name__ == '__main__':
    main()
