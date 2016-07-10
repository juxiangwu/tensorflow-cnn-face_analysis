import cv2
import os
import numpy as np

y_dataset = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

def readimages(path):
    x = []
    y = []

    dataList = os.listdir(path)

    for dataCount in range(len(dataList)):
        if (os.path.isdir(path + dataList[dataCount]) == False) :
            continue
        imageList = os.listdir(path + dataList[dataCount])
        print path + dataList[dataCount]

        for imageCount in range(len(imageList)):

            image = path + dataList[dataCount] + '/' + imageList[imageCount]
            feed = readimage(image)
            x.append(feed)
            y.append(y_dataset[int(dataList[dataCount][-2:])-1])

    print 'finish read images'
    return x, y

def readimage(path):
    img = cv2.imread(path)

    data = cv2.resize(img, (28, 28))

    return data

if __name__ == '__main__':
    pass
