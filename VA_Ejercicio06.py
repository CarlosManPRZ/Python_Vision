#Aplicar filtro: visión borrosa
#Filtro de promediado
#Algoritmo programado manualmente

#Lectura de imagen
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Lectura de imagen
img_org = cv.imread("Guacamaya.jpg")
img_org = cv.cvtColor(img_org, cv.COLOR_BGR2RGB)
m0,n0,p0 = img_org.shape
print(img_org.shape)
img_org = cv.resize(img_org,(int(n0/4),int(m0/4)))
m,n,p = img_org.shape
print(img_org.shape)


kd=3
kernel = np.ones((kd,kd),np.float32)/(kd*kd)
m1,n1 = kernel.shape 

img_mod = np.zeros((m,n,p),np.uint8)
img_mod2  = cv.filter2D(img_org,-1,kernel)
v=np.ones((1,m1),np.float32)
u=np.ones((m1,1),np.float32)
indice = np.uint8((m1-1)/2)
 


for j in range(indice ,m-indice):
    for i in range(indice,n-indice):
        a=j-indice
        b=j+indice+1
        c=i-indice
        d=i+indice+1
        img_mod[j,i,0] = np.uint8(v@(img_org[a:b,c:d,0]*kernel@u))
        img_mod[j,i,1] = np.uint8(v@(img_org[a:b,c:d,1]*kernel@u))
        img_mod[j,i,2] = np.uint8(v@(img_org[a:b,c:d,2]*kernel@u))
        

fig, ax= plt.subplots(1,3)
ax[0].imshow(img_org)
ax[0].set_title('Imagen original')
ax[1].imshow(img_mod)
ax[1].set_title('Filtro manual')
ax[2].imshow(img_mod2)
ax[2].set_title('Filtro Open CV')

##Comparacion de histogramas
#Histograma canal R
hist1 = cv.calcHist([img_mod], [0], None, [256], [0, 256])
hist2 = cv.calcHist([img_mod2], [0], None, [256], [0, 256])

#Histograma canal G
hist3 = cv.calcHist([img_mod], [1], None, [256], [0, 256])
hist4 = cv.calcHist([img_mod2], [1], None, [256], [0, 256])
#Histograma canal B
hist5 = cv.calcHist([img_mod], [2], None, [256], [0, 256])
hist6 = cv.calcHist([img_mod2], [2], None, [256], [0, 256])

fig2, ax2= plt.subplots(1,3)
fig2.suptitle('Comparación de Histogramas')
ax2[0].plot(hist1,'-')
ax2[0].plot(hist2,'--r')
ax2[0].set_title('Canal R')
ax2[1].plot(hist3)
ax2[1].plot(hist4,'--g')
ax2[1].set_title('Canal G')
ax2[2].plot(hist5)
ax2[2].plot(hist6,'--b')
ax2[2].set_title('Canal B')



plt.show()
