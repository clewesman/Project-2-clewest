import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that that takes an image and a
       transform, and computes the bounding box of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8 determine the outputs for this method.
    # raise Exception("TODO 8 in blend.py not implemented")
    rows,col,z = np.shape(img)
    transCoords = []
    transCoords.append(np.matmul(M,np.array([[0],[0],[1]])))    #get the corners post transofrmation and determine the largest x value
    transCoords.append(np.matmul(M,np.array([[rows-1],[0],[1]])))  
    transCoords.append(np.matmul(M,np.array([[0],[col-1],[1]])))
    transCoords.append(np.matmul(M,np.array([[rows-1],[col-1],[1]])))
    minX, maxX, minY, maxY = [0]*4
    for x in transCoords:
        if(x[0] < minX):
            minX = x[0]
        if(x[1] < minY):
            minY = x[1]
        if(x[0] > maxX):
            maxX = x[0]
        if(x[1] > maxY):
            maxY = x[1]
#right now prints 9 as lthe largest seen y value which comes from the last coord, should be 7 i guess 
#add divison by w and 

    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # convert input image to floats
    img = img.astype(np.float64) / 255.0

    # BEGIN TODO 10: Fill in this routine

    #run through al oop on blend width, run trhough loop on the heihtg of acc and use a formula 
    #really need a better explanation as to what to do here tbh 
    #acc contains whole image lecture 17 notes 


    raise Exception("TODO 10 in blend.py not implemented")
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11: fill in this routine
    img = []
    for i in range(acc.size[0]):
        for j in range(acc.size[1]):
            for k in range(3):
                alphaChan = acc[i,j,3]
                if(alphaChan != 0):
                    img[i,j,k] = acc[i,j,k]/alphaChan
            acc[i,j,3] = 1
                    #not sure the structure of hte pixels haha need to write that down so i divide properly 

    # raise Exception("TODO 11 in blend.py not implemented")
    # END TODO
    return (img * 255).astype(np.uint8)


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and returns useful information about the
       accumulated image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and
             transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all
             tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all
             tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        # this can (should) use the code you wrote for TODO 8
        
        #do i run through each image and transformation to find the overall min and max values? 
        #guessing once todo 8 is fixed i can then just copy and past code?
    for i in ipv:
        minX, maxX, minY, maxY = [0]*4
        minXtemp, minYtemp, maxXtemp, maxYtemp = imageBoundingBox(i.img,i.position)
        if(minX < minXtemp):
            minX = minXtemp
            minY = minYtemp
            maxX = maxXtemp
            maxY = maxYtemp


        # raise Exception("TODO 9 in blend.py not implemented")
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    """ Computes parameters for drift correction.
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         translation: transformation matrix so that top-left corner of accumulator image is origin
         width: Width of each image(assumption: all input images have same width)
       OUTPUT:
         x_init, y_init: coordinates in acc of the top left corner of the
            panorama with half the left image cropped out to match the right side
         x_final, y_final: coordinates in acc of the top right corner of the
            panorama with half the right image cropped out to match the left side
    """
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final

def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    if is360:
        # BEGIN TODO 12
        # 497P: you aren't required to do this. 360 mode won't work.

        # 597P: fill in appropriate entries in A to trim the left edge and
        # to take out the vertical drift:
        #   Shift it left by the correct amount
        #   Then handle the vertical drift - using a shear in the Y direction
        # Note: warpPerspective does forward mapping which means A is an affine
        # transform that maps accumulator coordinates to final panorama coordinates
        raise Exception("TODO 12 in blend.py not implemented")
        # END TODO 12

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage

