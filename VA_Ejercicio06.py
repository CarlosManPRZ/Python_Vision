#Aplicar filtro: escala de grises
#Algoritmo programado manualmente

#Lectura de imagen
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Lectura de imagen
img_org = cv.imread("Guacamaya.jpg")
img_org = cv.cvtColor(img_org, cv.COLOR_BGR2RGB)
img_org = cv.resize(img_org,(800,800))

m,n,p = img_org.shape
print(img_org.shape)


kernel = np.ones((15,15),np.float32)/(15*15)
m1,n1 = kernel.shape 

img_mod = np.zeros((m,n,p),np.uint8)
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
        

fig, ax= plt.subplots(1,2)
ax[0].imshow(img_org)
ax[1].imshow(img_mod)

plt.show()
