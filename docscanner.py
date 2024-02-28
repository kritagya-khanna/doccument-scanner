import cv2 as cv
import numpy as np

widthimg=480
heightimg=640

#---webcam setup
web=cv.VideoCapture(0)
web.set(3, 640)
web.set(4,480)
web.set(10, 200)




def preprocess(img):
    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgblur = cv.GaussianBlur(imggray,(5,5),1)
    imgcanny = cv.Canny(imgblur, 200, 200)
    kernel = np.ones((5,5))
    imgdialation=cv.dilate(imgcanny, kernel, iterations=2)
    imgthr=cv.erode(imgdialation,kernel, iterations=1)
    return imgthr


def getcontour(img):
    biggest = np.float32([])
    maxi = 0
    contours, hierarchy=cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for c in contours:
        area = cv.contourArea(c)
        if area>5:
            #cv.drawContours(imgcontour, c, -1,(255,255,0),3)
            peri=cv.arcLength(c, True)
            approx=cv.approxPolyDP(c,0.02*peri,True)
            if area > maxi and len(approx) == 4:
                biggest = approx
                maxi = area
    cv.drawContours(imgcontour, biggest, -1, (255, 255, 0), 20)
    return biggest
def reorder(mypoints):
    mypoints=mypoints.reshape((4,2))
    mypointsnew=np.zeros((4,1,2), np.int32)
    add = mypoints.sum(1)
    mypointsnew[0]=mypoints[np.argmin(add)]
    mypointsnew[3] = mypoints[np.argmax(add)]
    diff=np.diff(mypoints, axis=1)
    mypointsnew[1]= mypoints[np.argmin(diff)]
    mypointsnew[2] = mypoints[np.argmax(diff)]
    return mypointsnew
def getwrap(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [widthimg,0], [0,heightimg], [widthimg,heightimg]])
    matrix=cv.getPerspectiveTransform(pts1, pts2)
    imgoutput=cv.warpPerspective(img,matrix,(widthimg, heightimg))

    return imgoutput



while True:
    _,img = web.read()
    img = cv.resize(img,(widthimg,heightimg))
    imgthr = preprocess(img)
    imgcontour = img.copy()
    biggest = getcontour(imgthr)
    print(biggest)
    imgwarp = getwrap(img, biggest)


    cv.imshow("webcam", imgcontour)


    if cv.waitKey(1) & 0xFF== ord('q'):
        break
