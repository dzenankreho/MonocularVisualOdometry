from KITTISequenceLoader import KITTISequenceLoader
from Plotter import Plotter
import numpy as np
import cv2 as cv


class VisualOdometry:
    def __init__(self, sequenceLocation, plotProgress=False):
        self.plotProgress = plotProgress
        self.kittiLoader = KITTISequenceLoader(sequenceLocation)
        self.groundTruthPoses = self.kittiLoader.getAllGroundTruthPoses()
        self.K = self.kittiLoader.getIntrinsicCameraParameters()
        self.estimatedPoses = []
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
                imageShift.append((int(j * imageWidth / self.numOfPartsWidth),
                                   int(i * imageHeight / self.numOfPartsHeight)))
                imagesVec.append(
                    tuple([image[int(i * imageHeight / self.numOfPartsHeight):
                                 int((i + 1) * imageHeight / self.numOfPartsHeight),
                           int(j * imageWidth / self.numOfPartsWidth):
                                 int((j + 1) * imageWidth / self.numOfPartsWidth)]
                           for image in images]))

        return imagesVec, imageShift


    def findFeatureMatches2Frames(self, image1, image2):
        matchedKeypoints1 = []
        matchedKeypoints2 = []

        imagesVec, imageShift = self.splitImages(image1, image2)

        for cnt, images in enumerate(imagesVec):
            keypoints1, descriptors1 = self.orb.detectAndCompute(images[0], None)
            keypoints2, descriptors2 = self.orb.detectAndCompute(images[1], None)

            if isinstance(descriptors1, type(None)) or isinstance(descriptors2, type(None)):
                continue

            matches = self.matcher.match(descriptors1, descriptors2)

            for match in matches:
                matchedKeypoints1.append(np.array(keypoints1[match.queryIdx].pt)
                                         + imageShift[cnt])
                matchedKeypoints2.append(np.array(keypoints2[match.trainIdx].pt)
                                         + imageShift[cnt])

        return np.array(matchedKeypoints1), np.array(matchedKeypoints2)


    def findFeatureMatches3Frames(self, image1, image2, image3):
        matchedKeypoints1 = []
        matchedKeypoints2 = []
        matchedKeypoints3 = []

        imagesVec, imageShift = self.splitImages(image1, image2, image3)

        for cnt, images in enumerate(imagesVec):
            keypoints1, descriptors1 = self.orb.detectAndCompute(images[0], None)
            keypoints2, descriptors2 = self.orb.detectAndCompute(images[1], None)
            keypoints3, descriptors3 = self.orb.detectAndCompute(images[2], None)

            if isinstance(descriptors1, type(None)) or isinstance(descriptors2, type(None))\
                    or isinstance(descriptors3, type(None)):
                continue

            matches12 = self.matcher.match(descriptors1, descriptors2)
            matches23 = self.matcher.match(descriptors2, descriptors3)

            for match12 in matches12:
                for match23 in matches23:
                    if match12.trainIdx == match23.queryIdx:
                        matchedKeypoints1.append(np.array(keypoints1[match12.queryIdx].pt)
                                                 + imageShift[cnt])
                        matchedKeypoints2.append(np.array(keypoints2[match12.trainIdx].pt)
                                                 + imageShift[cnt])
                        matchedKeypoints3.append(np.array(keypoints3[match23.trainIdx].pt)
                                                 + imageShift[cnt])
                        # break

        return np.array(matchedKeypoints1), np.array(matchedKeypoints2), np.array(matchedKeypoints3)


    def run(self, firstFrameCnt=0, lastFrameCnt=None, useGroundTruthScale=True):
        if lastFrameCnt is None:
            lastFrameCnt = self.kittiLoader.getNumberOfFrames()

        prevPrevFrameCnt = None
        prevFrameCnt = firstFrameCnt
        currFrameCnt = prevFrameCnt + 1

        self.estimatedPoses.append(self.groundTruthPoses[firstFrameCnt])

        prevPrevFrame = None
        prevFrame = self.kittiLoader.getFrame(prevFrameCnt)
        currFrame = None

        if self.plotProgress:
            self.plotter.setupPlot()

        firstIter = True
        while True:
            if currFrameCnt == lastFrameCnt:
                break

            currFrame = self.kittiLoader.getFrame(currFrameCnt)

            matchedKeypoints1, matchedKeypoints2 = self.findFeatureMatches2Frames(prevFrame, currFrame)

            if self.plotProgress and firstIter:
                self.plotter.updatePlot(prevFrame, matchedKeypoints1)
                firstIter = False

            E, mask1 = cv.findEssentialMat(matchedKeypoints1, matchedKeypoints2, self.K,
                                           threshold=1, method=cv.RANSAC)
            _, R, t, mask2 = cv.recoverPose(E, matchedKeypoints1, matchedKeypoints2, self.K, mask=mask1)
            if not useGroundTruthScale and not isinstance(prevPrevFrameCnt, type(None)):
                scale = self.getRelativeScale(R, t, prevPrevFrame, prevFrame, currFrame)
            else:
                scale = self.kittiLoader.getGroundTruthScale(prevFrameCnt, currFrameCnt)
            t = t * scale

            if not np.logical_and.reduce((np.abs(t/scale) < 0.2) | (np.abs(t/scale) > 0.9))[0]:
                self.estimatedPoses.append(self.estimatedPoses[-1])
            else:
                self.estimatedPoses.append(self.estimatedPoses[-1]
                                           @ np.linalg.inv(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))))

                prevPrevFrameCnt = prevFrameCnt
                prevFrameCnt = currFrameCnt
                prevPrevFrame = prevFrame
                prevFrame = currFrame

            currFrameCnt += 1

            if self.plotProgress:
                self.plotter.updatePlot(currFrame, matchedKeypoints2)

        if self.plotProgress:
            self.plotter.plotResult()

        return np.array(self.estimatedPoses), self.groundTruthPoses


    def getRelativeScale(self, R, t, prevPrevFrame, prevFrame, currFrame):
        matchedKeypoints1, matchedKeypoints2, matchedKeypoints3 = \
            self.findFeatureMatches3Frames(prevPrevFrame, prevFrame, currFrame)

        matchedKeypoints1 = cv.undistortPoints(matchedKeypoints1, self.K, np.array([]))
        matchedKeypoints2 = cv.undistortPoints(matchedKeypoints2, self.K, np.array([]))
        matchedKeypoints3 = cv.undistortPoints(matchedKeypoints3, self.K, np.array([]))

        points3d12 = cv.triangulatePoints(self.estimatedPoses[-2][0:3, :],
                                          self.estimatedPoses[-1][0:3, :],
                                          matchedKeypoints1,
                                          matchedKeypoints2)

        points3d12 = np.transpose(points3d12)
        points3d12[:, 0] /= points3d12[:, 3]
        points3d12[:, 1] /= points3d12[:, 3]
        points3d12[:, 2] /= points3d12[:, 3]
        points3d12 = points3d12[:, 0:3]

        points3d23 = cv.triangulatePoints(self.estimatedPoses[-1][0:3, :],
                                          np.hstack((R, t)),
                                          matchedKeypoints2,
                                          matchedKeypoints3)

        points3d23 = np.transpose(points3d23)
        points3d23[:, 0] /= points3d23[:, 3]
        points3d23[:, 1] /= points3d23[:, 3]
        points3d23[:, 2] /= points3d23[:, 3]
        points3d23 = points3d23[:, 0:3]

        numOf3dPoints = np.shape(points3d12)[0]

        i = np.random.randint(0, numOf3dPoints, 300)
        j = np.random.randint(0, numOf3dPoints, 300)
        _, indices1, indices2 = np.intersect1d(i, j, return_indices=True)
        i = np.delete(i, indices1)
        j = np.delete(j, indices2)
        scales = np.linalg.norm(points3d12[i] - points3d12[j], axis=1) \
                 / np.linalg.norm(points3d23[i] - points3d23[j], axis=1)
        scales = scales[~np.isnan(scales) & (scales < 3) & (scales > 0.3)]

        return np.mean(scales)
