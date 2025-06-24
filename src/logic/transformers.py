from functools import lru_cache
from numba import njit, prange
import numpy as np
import math


# Константы для преобразования цветов
D65_X = 95.047
D65_Y = 100.000
D65_Z = 108.883
CIE_E = 216.0 / 24389.0
CIE_K = 24389.0 / 27.0

# Коэффициенты для Weighted Euclidean
WEIGHTED_EUCLIDEAN_WEIGHTS = np.array([0.299, 0.587, 0.114])

# Коэффициенты для Rec. ITU-R BT.2124 (2019)
BT2124_COEFFS = np.array([
    [0.70, 0.30, 0.00],
    [0.00, 1.00, 0.00],
    [0.00, 0.00, 1.00]
])


@lru_cache(maxsize=65536)
def rgb_to_lab_cached(rgb_tuple):
    """Кэшированное преобразование RGB в Lab"""
    return rgb_to_lab_numba(np.array(rgb_tuple))

@njit(fastmath=True)
def rgb_to_xyz_numba(rgb):
    """Преобразование RGB в XYZ с использованием Numba"""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

    # Inverse sRGB companding
    r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4

    # D65 reference white
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    return x * 100.0, y * 100.0, z * 100.0

@njit(fastmath=True)
def xyz_to_lab_numba(xyz):
    """Преобразование XYZ в Lab с использованием Numba"""
    x, y, z = xyz
    # D65 reference white
    x /= D65_X
    y /= D65_Y
    z /= D65_Z

    # Nonlinear transform
    x = x ** (1/3) if x > CIE_E else (CIE_K * x + 16) / 116
    y = y ** (1/3) if y > CIE_E else (CIE_K * y + 16) / 116
    z = z ** (1/3) if z > CIE_E else (CIE_K * z + 16) / 116

    lighting = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    return lighting, a, b

@njit(fastmath=True)
def rgb_to_lab_numba(rgb):
    """Преобразование RGB в Lab с использованием Numba"""
    xyz = rgb_to_xyz_numba(rgb)
    return xyz_to_lab_numba(xyz)

@njit(fastmath=True)
def ciede2000_numba_single(lab1, lab2):
    """Оптимизированная версия CIEDE2000 для одиночных цветов"""
    L1, a1, b1 = lab1[0], lab1[1], lab1[2]
    L2, a2, b2 = lab2[0], lab2[1], lab2[2]

    # Вычисление C'
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) * 0.5

    # Расчет G
    C_avg_pow7 = C_avg**7
    G = 0.5 * (1.0 - math.sqrt(C_avg_pow7 / (C_avg_pow7 + 6103515625.0)))  # 25^7 = 6103515625

    # a' вычисления
    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)

    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)

    # Углы h'
    h1_prime = 0.0 if (b1 == 0.0 and a1_prime == 0.0) else math.atan2(b1, a1_prime)
    if h1_prime < 0.0:
        h1_prime += 2.0 * math.pi

    h2_prime = 0.0 if (b2 == 0.0 and a2_prime == 0.0) else math.atan2(b2, a2_prime)
    if h2_prime < 0.0:
        h2_prime += 2.0 * math.pi

    # Разницы
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    # Вычисление delta_h_prime
    if C1_prime * C2_prime == 0.0:
        delta_h_prime = 0.0
    elif abs(h2_prime - h1_prime) <= math.pi:
        delta_h_prime = h2_prime - h1_prime
    elif h2_prime - h1_prime > math.pi:
        delta_h_prime = h2_prime - h1_prime - 2.0 * math.pi
    else:
        delta_h_prime = h2_prime - h1_prime + 2.0 * math.pi

    delta_H_prime = 2.0 * math.sqrt(C1_prime * C2_prime) * math.sin(delta_h_prime * 0.5)

    # Средние значения
    L_avg_prime = (L1 + L2) * 0.5
    C_avg_prime = (C1_prime + C2_prime) * 0.5

    # Вычисление h_avg_prime
    if C1_prime * C2_prime == 0.0:
        h_avg_prime = h1_prime + h2_prime
    else:
        if abs(h1_prime - h2_prime) <= math.pi:
            h_avg_prime = (h1_prime + h2_prime) * 0.5
        elif h1_prime + h2_prime < 2.0 * math.pi:
            h_avg_prime = (h1_prime + h2_prime + 2.0 * math.pi) * 0.5
        else:
            h_avg_prime = (h1_prime + h2_prime - 2.0 * math.pi) * 0.5

    # Весовые коэффициенты
    T = (1.0 - 0.17 * math.cos(h_avg_prime - 0.5235987755982988) +  # pi/6
         0.24 * math.cos(2.0 * h_avg_prime) +
         0.32 * math.cos(3.0 * h_avg_prime + 0.10471975511965977) -  # pi/30
         0.20 * math.cos(4.0 * h_avg_prime - 1.0995574287564276))    # 63*pi/180

    delta_theta = 0.5235987755982988 * math.exp(-((h_avg_prime * 57.29577951308232 - 275.0)/25.0)**2)

    R_C = 2.0 * math.sqrt(C_avg_prime**7 / (C_avg_prime**7 + 6103515625.0))
    S_L = 1.0 + (0.015 * (L_avg_prime - 50.0)**2) / math.sqrt(20.0 + (L_avg_prime - 50.0)**2)
    S_C = 1.0 + 0.045 * C_avg_prime
    S_H = 1.0 + 0.015 * C_avg_prime * T

    R_T = -math.sin(2.0 * delta_theta) * R_C

    # Итоговое delta_E
    delta_E = math.sqrt((delta_L_prime / S_L)**2 +
                       (delta_C_prime / S_C)**2 +
                       (delta_H_prime / S_H)**2 +
                       R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H))

    return delta_E

@njit(fastmath=True, parallel=True)
def ciede2000_numba_batch(lab_pixels, lab_palette):
    """Векторизованная версия CIEDE2000 для пакетной обработки"""
    n_pixels = lab_pixels.shape[0]
    n_colors = lab_palette.shape[0]
    distances = np.empty((n_pixels, n_colors), dtype=np.float32)

    for i in prange(n_pixels):
        for j in range(n_colors):
            distances[i, j] = ciede2000_numba_single(lab_pixels[i], lab_palette[j])

    return distances

@njit(fastmath=True)
def weighted_euclidean_distance(rgb1, rgb2, weights):
    """Вычисление взвешенного евклидова расстояния"""
    r_diff = rgb1[0] - rgb2[0]
    g_diff = rgb1[1] - rgb2[1]
    b_diff = rgb1[2] - rgb2[2]

    return math.sqrt(weights[0] * r_diff**2 +
                    weights[1] * g_diff**2 +
                    weights[2] * b_diff**2)

@njit(fastmath=True, parallel=True)
def weighted_euclidean_batch(pixels, palette, weights):
    """Векторизованная версия взвешенного евклидова расстояния"""
    n_pixels = pixels.shape[0]
    n_colors = palette.shape[0]
    distances = np.empty((n_pixels, n_colors), dtype=np.float32)

    for i in prange(n_pixels):
        for j in range(n_colors):
            distances[i, j] = weighted_euclidean_distance(pixels[i], palette[j], weights)

    return distances

@njit(fastmath=True)
def bt2124_transform(rgb):
    """Преобразование RGB в цветовое пространство Rec. ITU-R BT.2124"""
    r = rgb[0] / 255.0
    g = rgb[1] / 255.0
    b = rgb[2] / 255.0

    # Применяем матрицу преобразования
    c1 = BT2124_COEFFS[0,0] * r + BT2124_COEFFS[0,1] * g + BT2124_COEFFS[0,2] * b
    c2 = BT2124_COEFFS[1,0] * r + BT2124_COEFFS[1,1] * g + BT2124_COEFFS[1,2] * b
    c3 = BT2124_COEFFS[2,0] * r + BT2124_COEFFS[2,1] * g + BT2124_COEFFS[2,2] * b

    return c1, c2, c3

@njit(fastmath=True)
def bt2124_distance(rgb1, rgb2):
    """Вычисление расстояния в цветовом пространстве Rec. ITU-R BT.2124"""
    c1_1, c2_1, c3_1 = bt2124_transform(rgb1)
    c1_2, c2_2, c3_2 = bt2124_transform(rgb2)

    delta_c1 = c1_1 - c1_2
    delta_c2 = c2_1 - c2_2
    delta_c3 = c3_1 - c3_2

    return math.sqrt(delta_c1**2 + delta_c2**2 + delta_c3**2)

@njit(fastmath=True, parallel=True)
def bt2124_batch(pixels, palette):
    """Векторизованная версия расстояния Rec. ITU-R BT.2124"""
    n_pixels = pixels.shape[0]
    n_colors = palette.shape[0]
    distances = np.empty((n_pixels, n_colors), dtype=np.float32)

    for i in prange(n_pixels):
        for j in range(n_colors):
            distances[i, j] = bt2124_distance(pixels[i], palette[j])

    return distances