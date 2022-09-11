#Ejercicio 1 - Vision Artificial
#Transformacion Afin: Traslación,escalamiento y rotación
#Matriz de transformación calculada por libreria

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


Imagen_Original = cv.imread("edificio.png")

pts1 = np.float32([[174,194], [399, 99],
                       [176, 499], [370, 385]])
pts2 = np.float32([[0, 0], [399, 0],
                       [0, 500], [399, 500]])
     
# Apply Perspective Transform Algorithm
matrix = cv.getPerspectiveTransform(pts1, pts2)
result = cv.warpPerspective(Imagen_Original, matrix, (399, 500))

fig, axs = plt.subplots(1,2)
fig.suptitle('Perspectiva')

axs[0].imshow(Imagen_Original)
axs[0].set_title('Imagen original')
axs[0].set_xlabel('Dimensión de imagen: ' + str(Imagen_Original.shape))
axs[1].imshow(result)
axs[1].set_title('Imagen modificada')
axs[1].set_xlabel('Dimensión de imagen: ' + str(result.shape))
plt.show(block=False)
G=input('Escribe algo: ')
plt.show()
