#Primer ejemplo de python + Open CV en VSCode

#importar librerias
import numpy
import cv2
import matplotlib.pyplot as plt

#Crear objeto de imagen
guacamaya_im = cv2.imread("D:\Python\Guacamaya.jpg")
plt.title("Imagen cargada sin invertir colores BGR")
plt.xlabel ('Dimensión de imagen: ' + str(guacamaya_im.shape))
plt.imshow(guacamaya_im)
plt.show()