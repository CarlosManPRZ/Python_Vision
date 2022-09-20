#Aplicar filtro: Contraste (gain) y brillo(bias)
#Filtro u Operador punto a punto
#Algoritmo programado manualmente

#Lectura de imagen
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#definicion funcion saturacion
def sat(valor_pix):
    if valor_pix<=0:
        return 0
    elif valor_pix<= 255:  
        return valor_pix
    else:
        return 255

#definicino funcion filtro: filterGB
def filterGB(MatImg,Gain,Bias):
    mm,nn,pp =MatImg.shape
    Mat_result= np.zeros((mm,nn,pp),np.uint8)
    if pp>1:
        for j in range(0,mm):
            for i in range(0,nn):
                Mat_result[j,i,0] = sat( (Gain*MatImg[j,i,0])+Bias)
                Mat_result[j,i,1] = sat( (Gain*MatImg[j,i,1])+Bias)
                Mat_result[j,i,2] = sat( (Gain*MatImg[j,i,2])+Bias)
    return Mat_result

#Lectura de imagen
img_org = cv.imread("Guacamaya.jpg")
img_org = cv.cvtColor(img_org, cv.COLOR_BGR2RGB)
m0,n0,p0 = img_org.shape
print(img_org.shape)
reduccion_imagen =  4
img_org = cv.resize(img_org,(int(n0/reduccion_imagen),int(m0/reduccion_imagen)))
m,n,p = img_org.shape
print(img_org.shape)

#definicion de parametros y matrices
img_mod1 = np.zeros((m,n,p),np.uint8)
img_mod2 = np.zeros((m,n,p),np.uint8)

gain1 = 1
bias1 = 100
gain2 = 1.5
bias2=-100 

#Aplicacion de filtro           
img_mod1 = filterGB(img_org,gain1,bias1)
img_mod2 = filterGB(img_org,gain2,bias2)


fig, ax= plt.subplots(1,3)
fig.suptitle('Filtro lineal: '+ r'$g(i,j)=a\cdot f(i,j)+b,\, a>0$')
ax[0].imshow(img_org)
ax[0].set_title('Imagen original')
ax[1].imshow(img_mod1)
ax[1].set_title('gain=1,bias=100')
ax[2].imshow(img_mod2)
ax[2].set_title('gain=1.5,bias=-100')


##Comparacion de histogramas
#Histograma canal R 
hist1 = cv.calcHist([img_org], [0], None, [256], [0, 256])
hist2 = cv.calcHist([img_mod1], [0], None, [256], [0, 256])
hist7 = cv.calcHist([img_mod2], [0], None, [256], [0, 256])

#Histograma canal G - filtro 1
hist3 = cv.calcHist([img_org], [1], None, [256], [0, 256])
hist4 = cv.calcHist([img_mod1], [1], None, [256], [0, 256])
hist8 = cv.calcHist([img_mod2], [1], None, [256], [0, 256])
#Histograma canal B -
hist5 = cv.calcHist([img_org], [2], None, [256], [0, 256])
hist6 = cv.calcHist([img_mod1], [2], None, [256], [0, 256])
hist9 = cv.calcHist([img_mod2], [2], None, [256], [0, 256])

fig2, ax2= plt.subplots(1,3)
fig2.suptitle('Comparación de Histogramas con gain= '+str(gain1)+'y bias= '+str(bias1))
ax2[0].plot(hist1,'-')
ax2[0].plot(hist2,'--r')
ax2[0].legend(['Imagen original','Imagen modificada'])
ax2[0].set_title('Canal R')
ax2[1].plot(hist3)
ax2[1].plot(hist4,'--g')
ax2[1].set_title('Canal G')
ax2[1].legend(['Imagen original','Imagen modificada'])
ax2[2].plot(hist5)
ax2[2].plot(hist6,'--b')
ax2[2].set_title('Canal B')
ax2[2].legend(['Imagen original','Imagen modificada'])

fig3, ax3= plt.subplots(1,3)
fig3.suptitle('Comparación de Histogramas con gain= '+str(gain2)+'y bias= '+str(bias2))
ax3[0].plot(hist1,'-')
ax3[0].plot(hist7,'--r')
ax3[0].legend(['Imagen original','Imagen modificada'])
ax3[0].set_title('Canal R')
ax3[1].plot(hist3)
ax3[1].plot(hist8,'--g')
ax3[1].set_title('Canal G')
ax3[1].legend(['Imagen original','Imagen modificada'])
ax3[2].plot(hist5)
ax3[2].plot(hist9,'--b')
ax3[2].set_title('Canal B')
ax3[2].legend(['Imagen original','Imagen modificada'])

plt.show()
