#Shi-Tomasi corner detector
#Function implemented in OpenCV: 
# goodFeaturesToTrack

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#Image reading
img = cv.imread('edificio.png',cv.IMREAD_COLOR)
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img10 = img.copy()
img50 = img.copy()

#Corner detection
corners10 = cv.goodFeaturesToTrack(gray,10,0.01,10)
corners50 = cv.goodFeaturesToTrack(gray,50,0.01,10)
corners = cv.goodFeaturesToTrack(gray,0,0.01,10)

corners = np.int0(corners)
corners10 = np.int0(corners10)
corners50 = np.int0(corners50)

#corner_titles = ['corners','corners10','corners50']
#for j in corner_titles:
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),3,255,-1)

for i in corners10:
    x,y = i.ravel()
    cv.circle(img10,(x,y),3,255,-1)

for i in corners50:
    x,y = i.ravel()
    cv.circle(img50,(x,y),3,255,-1)

plt.figure()
plt.suptitle('Shi-Tomasi corner detector')
plt.subplot(131)
plt.imshow(img10)
plt.title('10 points')
plt.subplot(132)
plt.imshow(img50)
plt.title('50 points')
plt.subplot(133)
plt.imshow(img)
plt.title('default points')
#plt.savefig('shi-tomasi.png', bbox_inches='tight')
plt.show()
