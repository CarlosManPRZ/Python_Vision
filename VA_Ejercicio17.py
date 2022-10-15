#Transformada Wavelets
#Deteccion de bordes

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt
import pywt.data


# Load image
#original = pywt.data.camera()

imArray = cv2.imread('edificio.png',cv2.IMREAD_GRAYSCALE)
#convert to float
original =  np.float32(imArray)   
original /= 255

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'haar')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

plt.show()