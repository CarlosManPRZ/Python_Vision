#Filtro lineal: promediador
#Funcion utilizada: cv.boxFilter()

from locale import normalize
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Lectura en escala de grises
img = cv.imread('Guacamaya.jpg', cv.IMREAD_GRAYSCALE)

#Aplicacin de filtro promediador: Kernel 5x5
#Se activa la normalización
img_blur1 = cv.boxFilter(img,-1,(5,5),normalize=True)

#Aplicacion de filtro promediador: Kernel 15x15
img_blur2 = cv.boxFilter(img,-1,(15,15),normalize=True)

fig, ax= plt.subplots(1,3)
fig.suptitle('Filtro promediador: boxFilter normalizado')
plt.subplot(131),plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_blur1, cmap='gray'),plt.title('Kernel 5x5')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_blur2, cmap='gray'),plt.title('Kernel 15x15')
plt.xticks([]), plt.yticks([])


histr_org = cv.calcHist([img],[0],None,[256],[0,256])
histr_blur1 = cv.calcHist([img_blur1],[0],None,[256],[0,256])
histr_blur2 = cv.calcHist([img_blur2],[0],None,[256],[0,256])

plt.figure(2)
plt.title('Histogramas')
plt.plot(histr_org,'-k')
plt.plot(histr_blur1,'-b')
plt.plot(histr_blur2,'--r')
plt.legend(['Original','Kernel 5x5','Kernel 15x15'])


plt.show()
