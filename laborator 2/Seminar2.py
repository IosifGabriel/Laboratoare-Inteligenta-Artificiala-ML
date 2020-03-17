import numpy as np
from PIL import Image

image0 = np.load("images/car_0.npy")
image1 = np.load("images/car_1.npy")
image2 = np.load("images/car_2.npy")
image3 = np.load("images/car_3.npy")
image4 = np.load("images/car_4.npy")
image5 = np.load("images/car_5.npy")
image6 = np.load("images/car_6.npy")
image7 = np.load("images/car_7.npy")
image8 = np.load("images/car_8.npy")

listofimages = [image1, image2, image3, image4,
        image5, image6, image7, image8, image0]

def SumaPixeliAll():
    Sum = 0
    for i in range(len(listofimages)):
        Sum += np.sum(listofimages[i])
    return Sum
print("Suma tuturor pixelilor imaginilor este " + str(SumaPixeliAll()))

def ValoriPixeliImage():
    for i in range(len(listofimages)):
        print("Imaginea" + str(i) + " are valoarea " +
        str(np.sum(listofimages[i])) + " a pixelilor")

ValoriPixeliImage()

def IndexSumMax():
    max = -1
    index = -1
    for i in range(len(listofimages)):
        if(np.sum(listofimages[i])) > max:
            max = np.sum(listofimages[i])
            index = i
    return index

print("Indexul imaginii cu suma pixelilor este " + str(IndexSumMax()))

from skimage import io

def MeanImage():
    imagMean = np.zeros((400,600))
    for i in range(len(listofimages)):
        imagMean += listofimages[i]
    imagMean /= len(listofimages)
    return imagMean

io.imshow(MeanImage().astype(np.uint8))
io.show()

def DevStandard():
    return np.std(listofimages)

print("Deviatia standard este " + str(DevStandard()))

def Normalize():
    imagNormal = np.zeros((400,600))
    for i in range(len(listofimages)):
        imagNormal = (listofimages[i] - MeanImage()) / DevStandard()
        io.imshow(imagNormal.astype(np.uint8))
        io.show()

Normalize()


def ShowCroppedImages():
    imageCropped = np.zeros((100, 120))
    for i in range(len(listofimages)):
        imageCropped = listofimages[i][200:300,280:400]
        io.imshow(imageCropped.astype(np.uint8))
        io.show()

ShowCroppedImages()