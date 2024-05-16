import cv2
import numpy as np
import skimage.exposure

# load images
img1 = cv2.imread('k1.jpg')
img2 = cv2.imread('k2.png')
img3 = cv2.imread('k3.jpg')

# convert to gray
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)

# blur
blur1 = cv2.GaussianBlur(gray1, (0,0), sigmaX=6, sigmaY=6)
blur2 = cv2.GaussianBlur(gray2, (0,0), sigmaX=6, sigmaY=6)
blur3 = cv2.GaussianBlur(gray3, (0,0), sigmaX=6, sigmaY=6)

# morphology
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45,45))
morph1 = cv2.morphologyEx(blur1, cv2.MORPH_CLOSE, kernel)
morph2 = cv2.morphologyEx(blur2, cv2.MORPH_CLOSE, kernel)
morph3 = cv2.morphologyEx(blur3, cv2.MORPH_CLOSE, kernel)

# threshold
thresh1 = cv2.threshold(morph1, 0, 255, cv2.THRESH_OTSU)[1]
thresh2 = cv2.threshold(morph2, 0, 255, cv2.THRESH_OTSU)[1]
thresh3 = cv2.threshold(morph3, 0, 255, cv2.THRESH_OTSU)[1]

# get contours and filter on size
masked1 = gray1.copy()
meanval = int(np.mean(masked1))
contours = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:
    area = cv2.contourArea(cntr)
    if area > 500 and area < 50000:
        cv2.drawContours(masked1, [cntr], 0, (meanval), -1)

masked2 = gray2.copy()
meanval = int(np.mean(masked2))
contours = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:
    area = cv2.contourArea(cntr)
    if area > 500 and area < 50000:
        cv2.drawContours(masked2, [cntr], 0, (meanval), -1)

masked3 = gray3.copy()
meanval = int(np.mean(masked3))
contours = cv2.findContours(thresh3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:
    area = cv2.contourArea(cntr)
    if area > 500 and area < 50000:
        cv2.drawContours(masked3, [cntr], 0, (meanval), -1)

# stretch
minval = int(np.amin(masked1))
maxval = int(np.amax(masked1))
result1 = skimage.exposure.rescale_intensity(masked1, in_range=(minval,maxval), out_range=(0,255)).astype(np.uint8)
minval = int(np.amin(masked2))
maxval = int(np.amax(masked2))
result2 = skimage.exposure.rescale_intensity(masked2, in_range=(minval,maxval), out_range=(0,255)).astype(np.uint8)
minval = int(np.amin(masked3))
maxval = int(np.amax(masked3))
result3 = skimage.exposure.rescale_intensity(masked3, in_range=(minval,maxval), out_range=(0,255)).astype(np.uint8)

# save output
cv2.imwrite('xray1_stretched.png', result1)
cv2.imwrite('xray2_stretched.png', result2)
cv2.imwrite('xray3_stretched.png', result3)

# Display various images to see the steps
cv2.imshow('thresh1', thresh1)
cv2.imshow('thresh2', thresh2)
cv2.imshow('thresh3', thresh3)
cv2.imshow('result1', result1)
cv2.imshow('result2', result2)
cv2.imshow('result3', result3)
cv2.waitKey(0)
cv2.destroyAllWindows()