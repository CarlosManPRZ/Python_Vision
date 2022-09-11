import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Arreglos para los puntos de la perspectiva
#Puntos de origen
pts1 = np.float32([[0,0], [0,0],[0,0,], [0,0]])    
#Puntos de destino
pts2 = np.float32([[0,0], [0,0],[0,0,], [0,0]])    
#Variable global para indicar la posición en el arreglo de los puntos 
indice = 0

#Leer imagen original
img = cv.imread('edificio.png')
h1,w1,ch1 =img.shape
#Copia para manipulación
Imagen_Original = img.copy()

#Funcion para registrar las posiciones dentro de la imagen
def on_Event_LBUTTONDOWN(event,x,y,flags,param):
    global indice
    #Se activa cada vez que se hace un clic izquierdo (se presiona) con el mouse
    if event == cv.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x,y)
        #Los primeros 4 puntos corresponden al origen
        if indice <= 3:
            pts1[indice][:]=[x,y]
            #Agrega un círculo en la posición donde se hizo clic con el mouse  
            cv.circle(img,(x,y),10,(255,0,0),thickness=-1)
            #Agrega un texto indicando la posición donde se hizo clic con el mouse
            cv.putText(img,xy,(x,y),cv.FONT_HERSHEY_PLAIN,1.0,(0,0,0),thickness=1)
        #Los 4 puntos siguientes corresponden al destino    
        elif indice <=7:
             pts2[indice-4][:]=[x,y]  
             cv.circle(img,(x,y),10,(0,0,255),thickness=-1)
             cv.putText(img,xy,(x,y),cv.FONT_HERSHEY_PLAIN,1.0,(0,0,0),thickness=1)
        #Aumenta el indice cada vez que se hace clic
        indice = indice + 1       
        cv.imshow("image",img)
        
#Crea una ventana de nombre "image"
cv.namedWindow("image")
#Relaciona a la ventana una función del mose
cv.setMouseCallback("image",on_Event_LBUTTONDOWN)

while(1):
    cv.imshow("image",img)
    #Cuando se presiona ESCAPE se interrumpe la función        
    if cv.waitKey(0)&0xFF==27:
        break

#Cierra la ventana interactiva de la imagen
cv.destroyAllWindows()


 #Aplica la transformación de perspectiva a partir de los puntos seleccionados en la imagen original
matrix = cv.getPerspectiveTransform(pts1, pts2)
result = cv.warpPerspective(Imagen_Original, matrix, (w1, h1))

fig, axs = plt.subplots(1,2)
fig.suptitle('Transformación de Perspectiva')

axs[0].imshow(img)
axs[0].set_title('Imagen original')
axs[0].set_xlabel('Dimensión de imagen: ' + str(Imagen_Original.shape))
axs[1].imshow(result)
axs[1].set_title('Imagen modificada')
axs[1].set_xlabel('Dimensión de imagen: ' + str(result.shape))

plt.show()





