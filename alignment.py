import math
import random

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
            KeyPoint.pt holds a tuple of pixel coordinates (x, y)
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    #BEGIN TODO 2

    A = [] 
    for match in matches:   #runs trhough all the matches and construcst the A matrix
        img1_cords = match.queryIdx
        img2_cords = match.trainIdx
        (x1,y1) = f1[img1_cords].pt
        (x2,y2) = f2[img2_cords].pt

        A.append([x1,y1,1,0,0,0,-x2*x1,-x2*y1,-x2])   #formula to create the A matrix based on the two points in matches
        A.append([0,0,0,x1,y1,1,-y2*x1,-y2*y1,-y2])

    # Construct the A matrix that will be used to compute the homography
    # based on the given set of matches among feature sets f1 and f2.

    # raise Exception("TODO 2 in alignment.py not implemented")

    #END TODO

    if A_out is not None:
        A_out[:] = A

    x = minimizeAx(A) # find x that minimizes ||Ax||^2 s.t. ||x|| = 1


    H = np.eye(3) # create the homography matrix
    z = 0
    #BEGIN TODO 3
    #Fill the homography H with the correct values
    for i in range(3):   #fills in matrix with the valuse given by minimizeAx
        for j in range(3):
          H[i,j] = x[z]
          z+=1

    # raise Exception("TODO 3 in alignment.py not implemented")

    #END TODO

    return H

def minimizeAx(A):
    """ Given an n-by-m array A, return the 1-by-m vector x that minimizes
    ||Ax||^2 subject to ||x|| = 1.  This turns out to be the right singular
    vector of A corresponding to the smallest singular value."""
    return np.linalg.svd(A)[2][-1,:]

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslate) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call computeHomography.
    #This function should also call getInliers and, at the end,
    #least_squares_fit.
    
    bestInlierCount = -1

    #not sure how to add in homogrphy transformation portion 
    
    for i in range(nRANSAC):     #based on k value
        di = []
        if(m == "eTranslate"):   #determines wether to do a line or homogrophy 
            s = 1
            match = (random.choice(matches))
            (x1,y1) = f1[match.queryIdx].pt
            (x2,y2) = f1[match.trainIdx].pt
            x = x2-x1
            y = y2-y1
            tempM = np.array([[0,0,x][0,0,y][0,0,1]])
        else:
            s = 4
            for j in range(s):    # runs a random set of datapoints from matches based on S
              di.append(random.choice(matches))
            tempM = computeHomography(f1,f2,di,A_out=None)

        
        # not sure the usage of homgrophy funciton in it
        #fit model code goes here
          #either compute homogrophy or translation matrix asthe model

        inliers = getInliers(f1,f2,matches,tempM,RANSACthresh)  #gets inliers
        if(len(inliers) > bestInlierCount):      #checks for best inlier count so far
            bestInlierCount = len(inliers)
            M = tempM
            bestInliers = inliers
            #after finding the best model you can then use the best inliers to fit a final model  as a best step thing 
            # raise Exception("TODO 4 in alignment.py not implemented")
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''
   
    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        # Determine if the ith matched feature f1[matches[i].queryIdx], when
        # transformed by M, is within RANSACthresh of its match in f2.
        # If so, append i to inliers
        #TODO-BLOCK-BEGIN
        pt1 = np.array([[f1[matches[i].queryIdx].pt[0]] , [f1[matches[i].queryIdx].pt[1]] ,[1]])
        pt2 = np.array([[f2[matches[i].trainIdx].pt[0]] , [f2[matches[i].trainIdx].pt[1]] ,[1]])
        #add in division by w


        vect = np.dot(M,pt1)
        vect = vect / vect[2]
        dist = np.linalg.norm(np.subtract(vect ,pt2))

        if(np.linalg.norm(np.subtract(vect ,pt2)) < RANSACthresh):
            inlier_indices.append(i)   #should i  be appending the x,y values to inlier indices? 

# still recieving erros not sure where my math is wrong on this one 

        # raise Exception("TODO 5 in alignment.py not implemented")
        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0
        for i in inlier_indices:

            (x1,y1) = f1[matches[i].queryIdx].pt
            (x2,y2) = f2[matches[i].trainIdx].pt

            x = x2-x1
            y = y2-y1
            u+=x
            v+=y    
        avgXval = u/len(inlier_indices)
        avgYval = v/len(inlier_indices)

        M = np.array([[0,0,avgXval][0,0,avgYval][0,0,1]])
        #cvheck with a print statment 

        #not sure if my formula is right here 

        #BEGIN TODO 6 :Compute the average translation vector over all inliers.
        # Fill in the appropriate entries of M to represent the average
        # translation transformation.
        # raise Exception("TODO 6 in alignment.py not implemented")
        #END TODO

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers. This should call
        # computeHomography.
        totalmatches = []
        for i in inlier_indices:
            totalmatches.append(matches[i]) 

        M = computeHomography(f1,f2,totalmatches,A_out=None)

        # i believe that homogrophy is being performed properly 


        # raise Exception("TODO 7 in alignment.py not implemented")
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M
#bi linear interpolation (remap)