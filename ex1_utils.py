"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 207950577


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns in converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if representation == LOAD_GRAY_SCALE \
        else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final_img = (img_color - img_color.min()) / (img_color.max() - img_color.min())  # normalization
    return final_img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = cv2.imread(filename)
    if representation == LOAD_GRAY_SCALE:
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.gray()
    else:  # RGB image
        img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_color)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.311]])
    return np.dot(imgRGB, yiq_from_rgb.T.copy())


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.311]])
    rgb_from_yiq = np.linalg.inv(yiq_from_rgb)
    return np.dot(imgYIQ, rgb_from_yiq.T.copy())


# def calHist(img: np.ndarray) -> np.ndarray:
#     img_flat = img.flatten()
#     hist = np.zeros(256)
#
#     for pix in img_flat:
#         hist[pix] = hist[pix] + 1
#
#     return hist

# def calCumSum(arr: np.ndarray) -> np.ndarray:
#     cum_sum = np.zeros_like(arr)
#     cum_sum[0] = arr[0]
#     arr_len = len(arr)
#
#     for idx in range(1, arr_len):
#         cum_sum[idx] = arr[idx] + cum_sum[idx - 1]
#
#     return cum_sum


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """

    is_color = False
    if len(imgOrig.shape) == 3:  # if RGB image
        is_color = True
        yiqIm = transformRGB2YIQ(imgOrig)
        imgOrig = yiqIm[:, :, 0]
    imgOrig = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX)
    imgOrig = imgOrig.astype('uint8')
    histOrig = np.histogram(imgOrig.flatten(), bins=256)[0]
    cs = np.cumsum(histOrig)
    imgNew = cs[imgOrig]
    imgNew = cv2.normalize(imgNew, None, 0, 255, cv2.NORM_MINMAX)
    imgNew = imgNew.astype('uint8')
    histNew = np.histogram(imgNew.flatten(), bins=256)[0]
    if is_color:
        yiqIm[:, :, 0] = imgNew / (imgNew.max() - imgNew.min())
        imgNew = transformYIQ2RGB(yiqIm)
    return imgNew, histOrig, histNew


# def kmeans(pnts: np.ndarray, k: int, iter_num: int = 7) -> (np.ndarray, List[np.ndarray]):
#     if len(pnts.shape) > 1:
#         n,m = pnts.shape
#     else:
#         n = pnts.shape[0]
#         m = 1
#     pnts = pnts.reshape((n,m))
#     assign_array = np.random.randint(0, k, n)
#     centers = np.zeros((k, m))
#
#     assign_history = []
#     for it in range(iter_num):
#         for i, _ in enumerate(centers):
#             if sum(assign_array == i) < 1:
#                 continue
#             centers[i] = pnts[assign_array == i, :].mean(axis =0)
#
#         for i, p in enumerate(pnts):
#             center_dist = np.sqrt((centers -p) ** 2).sum(axis =1)
#             assign_array[i] = np.argmin(center_dist)
#         assign_history.append(assign_array.copy())
#     return centers, assign_history

def fix_q(z: np.array, image_hist: np.ndarray) -> np.ndarray:
    """
        Calculate the new q using wighted average on the histogram
        :param image_hist: the histogram of the original image
        :param z: the new list of centers
        :return: the new list of wighted average
    """
    q = [np.average(np.arange(z[k], z[k + 1] + 1), weights=image_hist[z[k]: z[k + 1] + 1]) for k in range(len(z) - 1)]
    return np.round(q).astype(int)

def fix_z(q: np.array) -> np.array:
    """
        Calculate the new z using the formula from the lecture.
        :param q: the new list of q
        :param z: the old z
        :return: the new z
    """
    z_new = np.array([round((q[i - 1] + q[i]) / 2) for i in range(1, len(q))]).astype(int)
    z_new = np.concatenate(([0], z_new, [255]))
    return z_new

def findBestCenters(histOrig: np.ndarray, nQuant: int, nIter: int) -> (np.ndarray, np.ndarray):
    """
            Finding the best nQuant centers for quantize the image in nIter steps or when the error is minimum
            :param histOrig: hist of the image (RGB or Gray scale)
            :param nQuant: Number of colors to quantize the image to
            :param nIter: Number of optimization loops
            :return: return all centers and they color selected to build from it all the images.
        """
    Z = []
    Q = []
    # head start, all the intervals are in the same length
    z = np.arange(0, 256, round(256 / nQuant))
    z = np.append(z, [255])
    Z.append(z.copy())
    q = fix_q(z, histOrig)
    Q.append(q.copy())
    for n in range(nIter):
        z = fix_z(q)
        if (Z[-1] == z).all():  # break if nothing changed
            break
        Z.append(z.copy())
        q = fix_q(z, histOrig)
        Q.append(q.copy())
    return Z, Q


def convertToImg(imOrig: np.ndarray, histOrig: np.ndarray, yiqIm: np.ndarray, arrayQuantize: np.ndarray) -> (
        np.ndarray, float):
    """
        Executing the quantization to the original image
        :return: returning the resulting image and the MSE.
    """
    imageQ = np.interp(imOrig, np.linspace(0, 1, 255), arrayQuantize)
    curr_hist = np.histogram(imageQ, bins=256)[0]
    err = np.sqrt(np.sum((histOrig.astype('float') - curr_hist.astype('float')) ** 2)) / float(
        imOrig.shape[0] * imOrig.shape[1])
    if len(yiqIm):  # if the original image is RGB
        yiqIm[:, :, 0] = imageQ / 255
        return transformYIQ2RGB(yiqIm), err
    return imageQ, err


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if len(imOrig.shape) == 3:
        imYIQ = transformRGB2YIQ(imOrig)
        imY = imYIQ[:, :, 0].copy()  # take only the y chanel
    else:
        imY = imOrig
    histOrig = np.histogram(imY.flatten(), bins=256)[0]
    Z, Q = findBestCenters(histOrig, nQuant, nIter)
    image_history = [imOrig.copy()]
    E = []
    for i in range(len(Z)):
        arrayQuantize = np.array([Q[i][k] for k in range(len(Q[i])) for x in range(Z[i][k], Z[i][k + 1])])
        q_img, e = convertToImg(imY, histOrig, imYIQ if len(imOrig.shape) == 3 else [], arrayQuantize)
        image_history.append(q_img)
        E.append(e)

    return image_history, E
