from kitti_sequence_loader import KITTISequenceLoader
from plotter import Plotter
import numpy as np
import cv2 as cv


class VisualOdometry:
    def __init__(self, sequenceLocation, plotProgress=False):
        self.plotProgress = plotProgress
        self.kittiLoader = KITTISequenceLoader(sequenceLocation)
        self.groundTruthPoses = self.kittiLoader.getAllGroundTruthPoses()
        self.estimatedPoses = [np.eye(4)]
        self.orb = cv.ORB_create(nfeatures=500, scoreType=cv.ORB_FAST_SCORE)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.numOfPartsWidth = 3
        self.numOfPartsHeight = 2
        
        if plotProgress:
            self.plotter = Plotter(self.groundTruthPoses, self.estimatedPoses)


    def splitImages(self, *images):
        imageHeight = np.shape(images[0])[0]
        imageWidth = np.shape(images[0])[1]
        imagesVec = []
        imageShift = []

        for i in range(self.numOfPartsHeight):
            for j in range(self.numOfPartsWidth):
                imageShift.append((int(i * imageHeight / self.numOfPartsHeight),
                                   int(j * imageWidth / self.numOfPartsWidth)))
                imagesVec.append(
                    tuple([image[int(i * imageHeight / self.numOfPartsHeight):
                                 int((i + 1) * imageHeight / self.numOfPartsHeight),
                           int(j * imageWidth / self.numOfPartsWidth):
                                 int((j + 1) * imageWidth / self.numOfPartsWidth)]
                           for image in images]))

        return imagesVec, imageShift


    def detectComputeAndMatchORB(self, image1, image2):
        matchedKeypoints1 = []
        matchedKeypoints2 = []

        imagesVec, imageShift = self.splitImages(image1, image2)

        for cnt, images in enumerate(imagesVec):
            keypoints1, descriptors1 = self.orb.detectAndCompute(images[0], None)
            keypoints2, descriptors2 = self.orb.detectAndCompute(images[1], None)

            if type(descriptors1) == type(None) or type(descriptors2) == type(None):
                continue

            matches = self.matcher.match(descriptors1, descriptors2)

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


    def run(self, firstFrameCnt=0, lastFrameCnt=None):
        K = self.kittiLoader.getIntrinsicCameraParameters()
        groundTruthPoses = self.kittiLoader.getAllGroundTruthPoses()

        if lastFrameCnt is None:
            lastFrameCnt = self.kittiLoader.getNumberOfFrames()

        prevFrameCnt = firstFrameCnt
        currFrameCnt = prevFrameCnt + 1

        if self.plotProgress:
            self.plotter.setupPlot()

        firstIter = True
        while True:
            # start = timer()
            if currFrameCnt == lastFrameCnt:
                break

            image1 = self.kittiLoader.getFrame(prevFrameCnt)
            image2 = self.kittiLoader.getFrame(currFrameCnt)

            matchedKeypoints1, matchedKeypoints2 = self.detectComputeAndMatchORB(image1, image2)

            if self.plotProgress and firstIter:
                self.plotter.updatePlot(image1, matchedKeypoints1)
                firstIter = False

            E, mask1 = cv.findEssentialMat(matchedKeypoints1, matchedKeypoints2, K, threshold=1, method=cv.RANSAC)
            _, R, t, mask2 = cv.recoverPose(E, matchedKeypoints1, matchedKeypoints2, K, mask=mask1)
            scale = self.kittiLoader.getGroundTruthScale(prevFrameCnt, currFrameCnt)
            t = t * scale

            if scale < 0.3:
                self.estimatedPoses.append(self.estimatedPoses[-1])
                currFrameCnt += 1
            else:
                self.estimatedPoses.append(self.estimatedPoses[-1]
                                           @ np.linalg.inv(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))))
                prevFrameCnt = currFrameCnt
                currFrameCnt += 1

            if self.plotProgress:
                self.plotter.updatePlot(image2, matchedKeypoints2)
            # print(timer()-start)

        if self.plotProgress:
            self.plotter.plotResult()

        return np.array(self.estimatedPoses), self.groundTruthPoses
