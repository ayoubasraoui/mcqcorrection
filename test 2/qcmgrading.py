# import cv2
# import numpy as np
# import util


# path = "scan1.jpg"
# widthImg=700
# heightImg=1000
# img = cv2.imread(path)


# #PREPROCCESSING
# #img= cv2.resize(img,(widthImg,heightImg))
# imgContours=img.copy()
# imgBiggestContours=img.copy()
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgBlur= cv2.GaussianBlur(imgGray,(5,5),1)
# imgCanny= cv2.Canny(imgBlur, 10,50)


# #FINDING ALL CONTOURS
# countours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(imgContours,countours,-1,(0,255,0),10)
# #FIND SHAPES
# rectCon=util.rectCountour(countours)
# biggestContour=util.getCornerPoints(rectCon[0])
# secondbiggestContour=util.getCornerPoints(rectCon[1])
# thirdbiggestContour=util.getCornerPoints(rectCon[2])
# forthbiggestContour=util.getCornerPoints(rectCon[3])
# #print(biggestContour)

# if biggestContour.size !=0 and secondbiggestContour !=0 and thirdbiggestContour!=0 and forthbiggestContour!=0:
#     cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),10)
#     cv2.drawContours(imgBiggestContours,secondbiggestContour,-1,(0,0,255),10)
#     cv2.drawContours(imgBiggestContours,thirdbiggestContour,-1,(255,10,10),10)
#     cv2.drawContours(imgBiggestContours,forthbiggestContour,-1,(255,255,0),10)


# imgBlank=np.zeros_like(img)
# imageArray=([img,imgGray,imgBlur,imgCanny],
#             [imgContours,imgBlank,imgBiggestContours,imgBlank,imgBlank])
# imgStacked= util.stackImages(imageArray, 0.5)


# if img is None:
#     print(f"Error: Unable to load image at {path}.")
# else:
#     cv2.imshow("Stacked images", imgStacked)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


################################################################


# import cv2
# import numpy as np
# import util

# path = "scan1.jpg"
# widthImg = 700
# heightImg = 1000
# img = cv2.imread(path)

# # PREPROCESSING
# # img = cv2.resize(img, (widthImg, heightImg))
# imgContours = img.copy()
# imgBiggestContours = img.copy()
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
# imgCanny = cv2.Canny(imgBlur, 10, 50)

# # FINDING ALL CONTOURS
# contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

# # FIND SHAPES
# rectCon = util.rectCountour(contours)

# # Safely get corner points, handle cases with fewer contours
# def get_valid_contour(rectCon, index):
#     if index < len(rectCon):
#         return util.getCornerPoints(rectCon[index])
#     return np.array([])  # Return an empty array if index is out of range

# biggestContour = get_valid_contour(rectCon, 0)
# # #secondbiggestContour = get_valid_contour(rectCon, 1)
# # #thirdbiggestContour = get_valid_contour(rectCon, 2)
# # #forthbiggestContour = get_valid_contour(rectCon, 3)

# # Check if contours are not empty
# def is_valid_contour(contour):
#     return contour.size > 0

# if is_valid_contour(biggestContour) : #and
#         ##is_valid_contour(secondbiggestContour) and
#         ##is_valid_contour(thirdbiggestContour) and
#         ##is_valid_contour(forthbiggestContour)):
#     cv2.drawContours(imgBiggestContours, [biggestContour], -1, (0, 255, 0), 10)
#     ##cv2.drawContours(imgBiggestContours, [secondbiggestContour], -1, (0, 0, 255), 10)
#     ##cv2.drawContours(imgBiggestContours, [thirdbiggestContour], -1, (255, 10, 10), 10)
#     ##cv2.drawContours(imgBiggestContours, [forthbiggestContour], -1, (255, 255, 0), 10)

# imgBlank = np.zeros_like(img)
# imageArray = ([img, imgGray, imgBlur, imgCanny],
#               [imgContours, imgBlank, imgBiggestContours, imgBlank, imgBlank])
# imgStacked = util.stackImages(imageArray, 0.5)

# if img is None:
#     print(f"Error: Unable to load image at {path}.")
# else:
#     cv2.imshow("Stacked images", imgStacked)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()





#######################################################################################

import cv2
import numpy as np
import util


path = "scan1.jpg"
widthImg = 700
heightImg = 1000
img = cv2.imread(path)

# PREPROCESSING
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

# FINDING ALL CONTOURS
countours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, countours, -1, (0, 255, 0), 10)

# FIND SHAPES
rectCon = util.rectCountour(countours)
biggestContour = util.getCornerPoints(rectCon[0]) if len(rectCon) > 0 else None
secondbiggestContour = util.getCornerPoints(rectCon[1]) if len(rectCon) > 1 else None
thirdbiggestContour = util.getCornerPoints(rectCon[2]) if len(rectCon) > 2 else None
forthbiggestContour = util.getCornerPoints(rectCon[3]) if len(rectCon) > 3 else None

if biggestContour is not None and secondbiggestContour is not None and thirdbiggestContour is not None and forthbiggestContour is not None:
    cv2.drawContours(imgBiggestContours, [biggestContour], -1, (0, 255, 0), 10)
    cv2.drawContours(imgBiggestContours, [secondbiggestContour], -1, (0, 0, 255), 10)
    cv2.drawContours(imgBiggestContours, [thirdbiggestContour], -1, (255, 10, 10), 10)
    cv2.drawContours(imgBiggestContours, [forthbiggestContour], -1, (255, 255, 0), 10)

imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgContours, imgBlank, imgBiggestContours, imgBlank, imgBlank])

# Specify the index of the image to resize (e.g., (1, 0) for imgContours in imageArray)
imgStacked = util.stackImages(imageArray, 0.5, resize_image_index=(1, 0), resize_factor=0.5)

if img is None:
    print(f"Error: Unable to load image at {path}.")
else:
    cv2.imshow("Stacked images", imgStacked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()