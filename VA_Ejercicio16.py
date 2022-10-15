#Transformada de fourier con OPENCV
#Se aplica la transformada de Fourier discreta
# a las imagenes de guacamaya y edificio

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Lectura de imagen en escala de grises
img_org1 = cv.imread('Guacamaya.jpg',cv.IMREAD_GRAYSCALE)
img_org2 = cv.imread('edificio.png',cv.IMREAD_GRAYSCALE)


 #Calculo de transformada de fourier discreta
dft1 = cv.dft(np.float32(img_org1),flags = cv.DFT_COMPLEX_OUTPUT)
dft2 = cv.dft(np.float32(img_org2),flags = cv.DFT_COMPLEX_OUTPUT)

#Mover el origen al centro de la imagen
dft_shift1 = np.fft.fftshift(dft1)
dft_shift2 = np.fft.fftshift(dft2)

 #Calculo de la magnitud
magnitude_spectrum1 = 20*np.log(cv.magnitude(dft_shift1[:,:,0],dft_shift1[:,:,1]))
magnitude_spectrum2 = 20*np.log(cv.magnitude(dft_shift2[:,:,0],dft_shift2[:,:,1]))


### Filtro pasa altas
rows, cols = img_org2.shape
crow,ccol = rows//2 , cols//2
# create a mask first, center square is 1, remaining all zeros
mask = np.ones((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 0
# apply mask and inverse DFT
fshift = dft_shift2*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv.idft(f_ishift)
img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.figure()
plt.subplot(121),plt.imshow(img_org2, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])


 #Graficas 
plt.figure() 
plt.subplot(121),plt.imshow(img_org1, cmap = 'gray')
plt.title('Imagen de entrada')
plt.subplot(122),plt.imshow(magnitude_spectrum1, cmap = 'gray')
plt.title('FFT')

plt.figure()
plt.subplot(121),plt.imshow(img_org2, cmap = 'gray')
plt.title('Imagen de entrada')
plt.subplot(122),plt.imshow(magnitude_spectrum2, cmap = 'gray')
plt.title('FFT')


plt.show()