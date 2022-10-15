#Comparación de filtros: Gausiano, Mediana 
#Se agregan ruidos gausianos y sal/pimienta a una imagen sin ruido
#Posteriormente se filtran las imagenes con ruido

import cv2 as cv
from cv2 import mean
from cv2 import GaussianBlur
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

#Lectura imagen sin ruido
img_noruido=cv.imread('MRI.jpg',cv.IMREAD_GRAYSCALE)
print(img_noruido.shape)

#Se añade ruido tipo gausiano a la imagen original
img_ruidoGauss = random_noise(img_noruido,mode='gaussian',seed=None,clip=True,mean=0.51,var=0.01)

#Se añade ruido sal y pimienta a la imagen original
img_ruidoSP = random_noise(img_noruido,mode='s&p',amount=0.3)

# The above function returns a floating-point image
# on the range [0, 1], thus we changed it to 'uint8'
# and from [0,255]
img_ruidoGauss = np.array(255*img_ruidoGauss, dtype = 'uint8')
img_ruidoSP = np.array(255*img_ruidoSP, dtype = 'uint8')

#Filtro Gaussiano a imagen con ruido gausiano
img_filtroGauss1 = cv.GaussianBlur(img_ruidoGauss,(5,5),0)

#Filtro Gaussiano a imagen con ruido sal y pimienta
img_filtroGauss2 = cv.GaussianBlur(img_ruidoSP,(5,5),0)

#Filtro de mediana a imagen con ruido gausiano
img_filtroMed1 = cv.medianBlur(img_ruidoGauss,5)

#Filtro de mediana a imagen con ruido sal y pimienta
img_filtroMed2 = cv.medianBlur(img_ruidoSP,5)


fig1, ax1= plt.subplots(1,3)
fig1.suptitle('Imagen con ruido gaussiano')
plt.subplot(131),plt.imshow(img_ruidoGauss, cmap='gray'),plt.title('Imagen con ruido gausiano')
plt.subplot(132),plt.imshow(img_filtroGauss1, cmap='gray'),plt.title('Filtro gausiano')
plt.subplot(133),plt.imshow(img_filtroMed1, cmap='gray'),plt.title('Filtro de mediana')

fig2, ax2= plt.subplots(1,3)
fig2.suptitle('Imagen con ruido sal y pimienta')
plt.subplot(131),plt.imshow(img_ruidoSP, cmap='gray'),plt.title('Imagen con ruido sal y pimienta')
plt.subplot(132),plt.imshow(img_filtroGauss2, cmap='gray'),plt.title('Filtro gausiano')
plt.subplot(133),plt.imshow(img_filtroMed2, cmap='gray'),plt.title('Filtro de mediana')

plt.show()
