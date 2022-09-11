import cv2 as cv #Importar OpenCV como cv
img = cv.imread("Paleta.png") #Leer imagen directo del archivo
imggs = cv.imread("Paleta.png",cv.IMREAD_GRAYSCALE)# Leer imagen de archivo y 
px = img[100,100]# Valor de pixel en posición 100,100 
print('Pixel    : ',px)#Imprimir valor pixel
dimensions = img.shape #Obtener dimensiones de la imagen
height = img.shape[0] #Altura
width = img.shape[1] #Ancho
channels = img.shape[2]#Canales
print('Image Dimension    : ',dimensions)#Imprimir dimensiones
print('Image Height       : ',height)#Imprimir altura
print('Image Width        : ',width)#imprimir ancho
print('Number of Channels : ',channels)#Imprimir canales
cv.imshow('imagen',img) #Mostrar imagen
cv.imshow('imagen GS',imggs) #Mostrar imagen
cv.waitKey(0) #Esperar a que el usuario cierre las ventanas