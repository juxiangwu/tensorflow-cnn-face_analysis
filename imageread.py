import cv2
import os
import numpy as np
import scipy.ndimage as nd

y_dataset = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

def getBestShift(img):
    cy,cx = nd.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def readimage(path):
    x = []
    y = []
    # y_dataset = np.loadtxt('classfication.txt', unpack=True)
    dataList = os.listdir(path)

    for dataCount in range(len(dataList)):
        imageList = os.listdir(path + dataList[dataCount])
        print path + dataList[dataCount]

        for imageCount in range(len(imageList)):
            image = path + dataList[dataCount] + '/' + imageList[imageCount]
            feed = doflat(image)
            x.append(feed)
            y.append(y_dataset[int(dataList[dataCount][-2:])-1])
            print y

    print 'finish read image'
    return x, y

def doflat(path):
    img = cv2.imread(path)

    data = cv2.resize(img, (28, 28))

    return data

if __name__ == '__main__':
    pass
