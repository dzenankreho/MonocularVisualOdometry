import numpy as np
import cv2 as cv

from kitti_sequence_loader import KITTISequenceLoader
from plotter import Plotter

loader = KITTISequenceLoader("KITTI07")
plotter = None

orb = cv.ORB_create(3000)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

FLANN_INDEX_LSH = 6
index_params= dict(algorithm=FLANN_INDEX_LSH,
                   table_number=12, # 12 6
                   key_size=20, # 20 12
                   multi_probe_level=2) #2 1
flann = cv.FlannBasedMatcher(index_params)


def detectComputeAndMatchORB(image1, image2):
    imageHeight = np.shape(image1)[0]
    imageWidth = np.shape(image1)[1]

    imagesVec = []
    imageShift = []
    subimageWidthRatio = 3
    subimageHeightRatio = 2
    for i in range(subimageHeightRatio):
        for j in range(subimageWidthRatio):
            imageShift.append((int(i*imageHeight/subimageHeightRatio),
                               int(j*imageWidth/subimageWidthRatio)))
            imagesVec.append((image1[int(i*imageHeight/subimageHeightRatio):int((i+1)*imageHeight/subimageHeightRatio),
                                     int(j*imageWidth/subimageWidthRatio):int((j+1)*imageWidth/subimageWidthRatio)],
                              image2[int(i*imageHeight/subimageHeightRatio):int((i+1)*imageHeight/subimageHeightRatio),
                                     int(j*imageWidth/subimageWidthRatio):int((j+1)*imageWidth/subimageWidthRatio)]))

    matchedKeypoints1 = []
    matchedKeypoints2 = []

    for cnt, images in enumerate(imagesVec):
        keypoints1, descriptors1 = orb.detectAndCompute(images[0], None)
        keypoints2, descriptors2 = orb.detectAndCompute(images[1], None)

        if type(descriptors1) == type(None) or type(descriptors2) == type(None):
            continue

        matches = bf.match(descriptors1, descriptors2)
        # matches = sorted(matches, key=lambda x: x.distance)

        for match in matches:
            (x1, y1) = keypoints1[match.queryIdx].pt
            (x2, y2) = keypoints2[match.trainIdx].pt

            y1 += imageShift[cnt][0]
            y2 += imageShift[cnt][0]
            x1 += imageShift[cnt][1]
            x2 += imageShift[cnt][1]

            matchedKeypoints1.append((x1, y1))
            matchedKeypoints2.append((x2, y2))

    return np.array(matchedKeypoints1), np.array(matchedKeypoints2)


def runVO(firstFrameCnt = 0, lastFrameCnt = None):
    estimatedPoses = [np.eye(4)]
    K = loader.getIntrinsicCameraParameters()
    groundTruthPoses = loader.getAllGroundTruthPoses()

    if lastFrameCnt is None:
        lastFrameCnt = loader.getNumberOfFrames()

    prevFrameCnt = firstFrameCnt
    currFrameCnt = prevFrameCnt + 1

    estimatedPoses[0] = groundTruthPoses[prevFrameCnt]

    global plotter
    plotter = Plotter(groundTruthPoses, estimatedPoses)
    plotter.setupPlot()

    firstIter = True
    while True:
        if currFrameCnt == lastFrameCnt:
            break

        image1 = loader.getFrame(prevFrameCnt)
        image2 = loader.getFrame(currFrameCnt)

        matchedKeypoints1, matchedKeypoints2 = detectComputeAndMatchORB(image1, image2)
        # print(np.shape(matchedKeypoints1))
        if firstIter:
            plotter.updatePlot(image1, matchedKeypoints1)
            firstIter = False

        E, mask1 = cv.findEssentialMat(matchedKeypoints1, matchedKeypoints2, K, threshold=1, method=cv.RANSAC)
        _, R, t, mask2 = cv.recoverPose(E, matchedKeypoints1, matchedKeypoints2, K, mask=mask1)
        groundTruthScale = loader.getGroundTruthScale(prevFrameCnt, currFrameCnt)
        t = t * groundTruthScale

        if groundTruthScale < 0.25:
            estimatedPoses.append(estimatedPoses[-1])
            currFrameCnt += 1
        else:
            estimatedPoses.append(estimatedPoses[-1]
                                  @ np.linalg.inv(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))))
            prevFrameCnt = currFrameCnt
            currFrameCnt += 1

        plotter.updatePlot(image2, matchedKeypoints2)

    return np.array(estimatedPoses)


estimatedPoses = runVO()
plotter.plotResult()
