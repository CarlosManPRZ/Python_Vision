#Ecualización de histograma adaptable
#Contraste de ecualización adaptable del histograma
#CLAHE
#Imagen original: Guacamaya brillo horizontal

#Lectura de imagen
from typing import ChainMap
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Lectura de imagen
img_org_BGR = cv.imread("Guacamaya_briloH.jpg")
img_org = cv.cvtColor(img_org_BGR, cv.COLOR_BGR2RGB)
m0,n0,p0 = img_org.shape
print(img_org.shape)
color = ('r','g','b')

fig, ax= plt.subplots(1,3)
#Ecualizacion adaptable
img_eqAd = np.zeros((m0,n0,p0),np.uint8)
clahe= cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
for channel,col in enumerate(color):
    img_eqAd[:,:,channel]= clahe.apply(img_org[:,:,channel])
    histr_eqAd = cv.calcHist([img_eqAd],[channel],None,[256],[0,256])
    ax[2].plot(histr_eqAd, color = col)


#Ecualizar canales -metodo normal
img_eq = np.zeros((m0,n0,p0),np.uint8)
for channel,col in enumerate(color):
    img_eq[:,:,channel] = cv.equalizeHist(img_org[:,:,channel])
    histr_eq = cv.calcHist([img_eq],[channel],None,[256],[0,256])
    ax[1].plot(histr_eq,color = col)

for channel,col in enumerate(color):
    histr_org = cv.calcHist([img_org],[channel],None,[256],[0,256])
    ax[0].plot(histr_org,color = col)


ax[0].set_title('Histograma original')
ax[1].set_title('Histograma ecualizacion normal')
ax[2].set_title('Histograma ecualizacion adaptable')


fig2, ax2=plt.subplots(1,3) 
ax2[0].imshow(img_org)
ax2[0].set_title('Imagen original')
ax2[1].imshow(img_eq)
ax2[1].set_title('Ecualizacion normal')
ax2[2].imshow(img_eqAd)
ax2[2].set_title('Ecualizacion adaptable')

plt.show()