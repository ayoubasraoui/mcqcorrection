import cv2
import numpy as np


# def stackImages(imgArray, scale, labels=[]):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
    
#     # Check if the rows are available as lists
#     rowsAvailable = isinstance(imgArray[0], list)
    
#     if rowsAvailable:
#         # Resize images and convert grayscale to BGR
#         for x in range(rows):
#             for y in range(cols):
#                 imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2:
#                     imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        
#         # Find width and height from the first image
#         height, width = imgArray[0][0].shape[:2]
        
#         # Create blank image for stacking
#         imageBlank = np.zeros((height, width, 3), np.uint8)
        
#         # Stack images horizontally in each row
#         hor = [imageBlank] * rows
#         for x in range(rows):
#             hor[x] = np.hstack(imgArray[x])
        
#         # Stack rows vertically
#         ver = np.vstack(hor)
#     else:
#         # Resize images and convert grayscale to BGR
#         imgArray = [cv2.resize(img, (0, 0), None, scale, scale) for img in imgArray]
#         imgArray = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img for img in imgArray]
        
#         # Stack images horizontally
#         hor = np.hstack(imgArray)
#         ver = hor

#     # Add labels if provided
#     if len(labels) != 0:
#         eachImgWidth = int(ver.shape[1] / cols)
#         eachImgHeight = int(ver.shape[0] / rows)
#         for d in range(rows):
#             for c in range(cols):
#                 cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d), 
#                               (c * eachImgWidth + len(labels[d][c]) * 13 + 27, 30 + eachImgHeight * d), 
#                               (255, 255, 255), cv2.FILLED)
#                 cv2.putText(ver, labels[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20), 
#                             cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    
#     return ver





def stackImages(imgArray, scale, labels=[], resize_image_index=None, resize_factor=0.5):
    rows = len(imgArray)
    cols = len(imgArray[0])
    
    # Determine the dimensions for resizing
    if resize_image_index is not None:
        resize_img = imgArray[resize_image_index[0]][resize_image_index[1]]
        new_width = int(resize_img.shape[1] * resize_factor)
        new_height = int(resize_img.shape[0] * resize_factor)
    else:
        # Find the smallest width and height among all images
        min_width = min(imgArray[i][j].shape[1] for i in range(rows) for j in range(cols))
        min_height = min(imgArray[i][j].shape[0] for i in range(rows) for j in range(cols))
        new_width = int(min_width * scale)
        new_height = int(min_height * scale)
    
    resized_images = []
    for i in range(rows):
        row_images = []
        for j in range(cols):
            img = imgArray[i][j]
            if resize_image_index and (i, j) == resize_image_index:
                resized_img = cv2.resize(img, (new_width, new_height))
            else:
                resized_img = cv2.resize(img, (new_width, new_height))
            if len(resized_img.shape) == 2:  # Convert grayscale to BGR
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
            row_images.append(resized_img)
        resized_images.append(row_images)
    
    # Create a blank image with the new size
    imageBlank = np.zeros((new_height, new_width, 3), np.uint8)
    hor = [imageBlank] * rows
    
    # Stack images horizontally
    for x in range(rows):
        hor[x] = np.hstack(resized_images[x])
    
    # Stack images vertically
    ver = np.vstack(hor)
    
    # Add labels if provided
    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(rows):
            for c in range(cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d), 
                              (c * eachImgWidth + len(labels[d][c]) * 13 + 27, 30 + eachImgHeight * d), 
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, labels[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    
    return ver








def rectCountour(countours):
    rectCon=[]
    
    for i in countours:
        area=cv2.contourArea(i)
        #print("Area",area)
        if area>50 :
            peri=cv2.arcLength(i,True)
            approx=cv2.approxPolyDP(i,0.02*peri,True)
            #print("Corner Points",len(approx))
            if len(approx)==6:
                rectCon.append(i)
    rectCon=sorted(rectCon,key=cv2.contourArea,reverse=True)
    
    return rectCon

def getCornerPoints(cont):
        peri=cv2.arcLength(cont,True)
        approx=cv2.approxPolyDP(cont,0.02*peri,True)
        return approx