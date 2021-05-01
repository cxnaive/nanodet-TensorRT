import numpy as np
import cv2
import random


# def random_brightness(img, delta):
#     img += random.uniform(-delta, delta)
#     return img

#Gamma trans
def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

def random_gamma(img, gamma_low, gamme_up):
    gamma = random.uniform(gamma_low, gamme_up)
    return gamma_trans(img,gamma)


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def normalize(meta, mean, std):
    img = meta['img'].astype(np.float32)
    mean = np.array(mean, dtype=np.float64).reshape(1, -1)
    stdinv = 1 / np.array(std, dtype=np.float64).reshape(1, -1)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    meta['img'] = img
    return meta


def _normalize(img, mean, std):
    img = img.astype(np.float32)
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    return img


def color_aug_and_norm(meta, kwargs):
    img = meta['img'] #.astype(np.float32) / 255
    if 'gamma' in kwargs and random.randint(0, 1):
        img = random_gamma(img, *kwargs['gamma'])
    #print(kwargs)
    if 'hsv_h' in kwargs and 'hsv_s' in kwargs and 'hsv_v' in kwargs:
        hgain=kwargs['hsv_h']
        sgain=kwargs['hsv_s']
        vgain=kwargs['hsv_v']
        augment_hsv(img,hgain,sgain,vgain)
    # if 'brightness' in kwargs and random.randint(0, 1):
    #     hsv_img = random_brightness(hsv_img, kwargs['brightness'])

    # if 'saturation' in kwargs and random.randint(0, 1):
    #     hsv_img = random_saturation(hsv_img, *kwargs['saturation'])

    # if 'contrast' in kwargs and random.randint(0, 1):
    #     img = random_contrast(img, *kwargs['contrast'])
    # cv2.imshow('trans', img)
    # cv2.waitKey(0)
    # img = _normalize(img, *kwargs['normalize'])
    meta['img'] = img.astype(np.float32) / 255.0
    return meta


