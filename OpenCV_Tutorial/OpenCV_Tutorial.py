
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir(r'C:\Users\Goo\Desktop\Personal Learning\OpenCV_Tutorial')

from utils.mytool import stackImages,getContours
#%% 1 - Displaying image
img = cv2.imread('resources/lena.png')

cv2.imshow('output', img)
cv2.waitKey(0)

#%% 1 - Displaying video

cap = cv2.VideoCapture("resources/Wind turbine.mp4")

while True:
    success, img = cap.read()
    cv2.imshow("Video",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
#%% 1 - Displaying webcam

cap = cv2.VideoCapture(0)
cap.set(3,640) # cv2.CAP_PROP_FRAME_WIDTH == 3
cap.set(4,480)
cap.set(10,100)

while True:
    success, img = cap.read()
    cv2.imshow("Video",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
#%% 2 - Converting to Gray Image/ Blurring / Edge detector(Canny)/ Dilation/ Erosion

img = cv2.imread("resources/lena.png")
kernel = np.ones((5,5),dtype=np.uint8)

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 0)
imgCanny = cv2.Canny(img,150,200)
imgDilate = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgDilate, kernel, iterations=1)
cv2.imshow('Gray',imgGray)
cv2.imshow('Blur',imgBlur)
cv2.imshow('Canny',imgCanny)
cv2.imshow('Dilation',imgDilate)
cv2.imshow('Erosion',imgEroded)
cv2.waitKey(0)
#%% 3 - Resizing and Cropping

img = cv2.imread('resources/house.jpg')
print(img.shape)

imgResize = cv2.resize(img, (400,300))
cv2.imshow('resized',imgResize)

imgCropped = imgResize[0:200,0:300]
cv2.imshow('Cropped',imgCropped)
cv2.waitKey(0)
#%% 4 - Shape and Texts

img = np.zeros([512,512,3], dtype=np.uint8)
img = img.astype(dtype=np.float32)
cv2.imshow('Black',img)
# img[:] = 255,0,0
# cv2.imshow('Blue',img)
cv2.line(img, (0,0), (300,300), color= (0,255,0), thickness=3)
# cv2.line(img, (0,0), (img.shape[1],img.shape[0]), color= (0,255,0), thickness=3)
cv2.imshow('line',img)

cv2.rectangle(img, (0,0), (250,350), color = (0,0,255), thickness=2)
# cv2.rectangle(img, (0,0), (250,350), (0,0,255), cv2.FILLED)
cv2.imshow('Rectangle',img)

cv2.circle(img, (400,50), 30, (255,255,0), 5)
cv2.imshow('Circle',img)

cv2.putText(img, "OpenCV", (300,100), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0,150,0), 1) 
cv2.imshow('text added',img)

cv2.waitKey(0)
#%% 5 - Warp perspective

img = cv2.imread('resources/house.jpg')

width, height = 150, 300
# Points were manually acquired by opening image with paint app 
lt = [582, 820]
rt = [803, 773]
lb = [581, 1136]
rb = [871, 1087]
pts1 = np.float32([lt, rt, lb, rb])
pts2 = np.float32([[0,0], [width,0], [0,height], [width,height]])

matrix = cv2.getPerspectiveTransform(pts1,pts2) # transformation matrix
imgOutput = cv2.warpPerspective(img, matrix, (width,height))

cv2.imshow('Image',img)
cv2.imshow('Wrap Perspective',imgOutput)
cv2.waitKey(0)
#%% 6 - Joining Images

img = cv2.imread('resources/lena.png')

imgHor = np.hstack((img,img))
imgVer = np.vstack((img,img))
cv2.imshow("Horizontal", imgHor)
cv2.imshow("Vertical", imgVer)
if cv2.waitKey(0)==27:
    cv2.destroyAllWindows()
#%% 7 - Color detection

def empty(a):
    pass

cv2.namedWindow(winname = "TrackBars")

cv2.resizeWindow(winname = "TrackBars",
                width = 640,
                height = 240)

cv2.createTrackbar("Hue Min",
                   "TrackBars",
                   85, # initial value
                   179, # maximum hue value (For OpenCV, it is 0-179)
                   empty) # need to define a function which gets called on
cv2.createTrackbar("Hue Max", "TrackBars", 102, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 35, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 142, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    
    img = cv2.imread('resources/house.jpg')
    img = cv2.resize(img, (600,400))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgRes = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('Image',img)
    cv2.imshow('HSV',imgHSV)
    cv2.imshow('Mast',mask)
    cv2.imshow('Result',imgRes)
    cv2.waitKey(1)
#%% 8 - Contours & Shape detection
    
img = cv2.imread('resources/shapes.png')

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
imgCanny = cv2.Canny(imgBlur, 50,50)
imgBlank = np.zeros_like(img)
imgContour = getContours(img, imgCanny)
imgStack = stackImages(0.6, [[img, imgGray, imgBlur],
                             [imgCanny, imgContour, imgBlank]])


cv2.imshow('Stack',imgStack)
if cv2.waitKey(0)==27:
    cv2.destroyAllWindows()
#%% 9 - Face detection

faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
img = cv2.imread('resources/lena.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

cv2.imshow('Result',img)
if cv2.waitKey(0)==27:
    cv2.destroyAllWindows()