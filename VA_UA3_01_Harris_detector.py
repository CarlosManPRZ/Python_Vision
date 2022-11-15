#Detector de esquinas de Harris
#Sin usar librería propia de OpenCV
#Comparación final con la función pre-programada

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Lectura de imagen
img1 = cv.imread('edificio.png')
img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
img2 = img1.copy()
#Conversion a escala de grises
img_gray1 = cv.cvtColor(img1,cv.COLOR_RGB2GRAY)
img_gray2 = img_gray1.copy()

#Derivadas con Sobel
grad_x = cv.Sobel(img_gray1, cv.CV_64F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
grad_y = cv.Sobel(img_gray1, cv.CV_64F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

#Multiplicacion de derivadas
Ix2 = grad_x**2
Iy2 = grad_y**2
Ixy = grad_x*grad_y

#Suma de productos con filtro promediador
kernel1 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]],np.float32)*(1/9)
 
Sx2 = cv.filter2D(src=Ix2, ddepth=-1, kernel=kernel1)
Sy2 = cv.filter2D(src=Iy2, ddepth=-1, kernel=kernel1)
Sxy = cv.filter2D(src=Ixy, ddepth=-1, kernel=kernel1)

#Matriz para indicador de Harris
[m,n]=Sx2.shape
R_matrix=np.zeros((m,n))

#Obtencion de indicador de Harris
umbral = 10000000000
for i in range(1,n):
    for j in range(1,m):
        H=np.array([ [Sx2[j,i],Sxy[j,i]], [Sxy[j,i],Sy2[j,i]] ])
        r=np.linalg.det(H) - (0.04*((np.matrix.trace(H))**2))
        if r>umbral:
            R_matrix[j,i]=r
            

#Non-maximal supression 
#El resultado se dilata para obtener máximos
dst1 = cv.dilate(R_matrix,None)
#Threshold for an optimal value, it may vary depending on the image.
img1[dst1>0.01*dst1.max()]=[255,0,0]


#Deteccion por funcion de OpenCV
img_gray2 = np.float32(img_gray2)
#Detector de harris con nucleo de 3x3,k=0.04 y matrix correlacion 2x2
dst2 = cv.cornerHarris(img_gray2,2,3,0.04)
#result is dilated for marking the corners, not important
dst2 = cv.dilate(dst2,None)
# Threshold for an optimal value, it may vary depending on the image.
img2[dst2>0.01*dst2.max()]=[0,0,255]

#Graficas de resultados
plt.figure()
plt.suptitle('Harris corner detector')
plt.subplot(121)
plt.imshow(img1)
plt.title('Manually implemented')
plt.subplot(122)
plt.imshow(img2)
plt.title('OpenCV implemented')
#plt.savefig('harris_corner01.png', bbox_inches='tight')
plt.show()
