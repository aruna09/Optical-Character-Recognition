import cv2
import numpy as np

img = cv2.imread("image30.png")

img = cv2.resize(img, (1000, 500))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
flag, thresh = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)

im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imshow("img", img)

edges = cv2.Canny(gray_img,50,150,apertureSize = 3)

minLineLength = 10
maxLineGap = 2

lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('horizon_detected.jpg',img)

cv2.waitKey(0)
cv2.destroyAllWindows()