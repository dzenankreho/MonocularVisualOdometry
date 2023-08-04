from KITTISequenceLoader import KITTISequenceLoader
from Plotter import Plotter
import numpy as np
import cv2 as cv
import warnings


class VisualOdometry:
    def __init__(self, sequenceLocation, plotProgress=False):
        self.plotProgress = plotProgress
        self.kittiLoader = KITTISequenceLoader(sequenceLocation)
        self.groundTruthPoses = self.kittiLoader.getAllGroundTruthPoses()
        self.K = self.kittiLoader.getIntrinsicCameraParameters()
        self.estimatedPoses = []
        self.orb = cv.ORB_create(nfeatures=500, scoreType=cv.ORB_FAST_SCORE)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.numOfPartsWidth = 1
        self.numOfPartsHeight = 1
        self.prevFrameCnt = None
        self.currFrameCnt = None
        self.prevPrevFrame = None
        self.prevFrame = None
        self.currFrame = None
        self.currFrameKeypoints = None

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

        return np.array(matchedKeypoints1), np.array(matchedKeypoints2), np.array(matchedKeypoints3)


    def run(self, firstFrameCnt=0, lastFrameCnt=None, useGroundTruthScale=True, motionEstimation='2d-2d'):
        if motionEstimation.lower() == '2dto2d' or motionEstimation.lower() == '3dto2d':
            motionEstimation = motionEstimation.lower().replace('to', '-')

        if motionEstimation != '2d-2d' and motionEstimation != '3d-2d':
            raise ValueError("Invalid motion estimation select variable")

        if lastFrameCnt is None:
            lastFrameCnt = self.kittiLoader.getNumberOfFrames()

        self.prevFrameCnt = firstFrameCnt
        self.currFrameCnt = self.prevFrameCnt + 1
        self.estimatedPoses.append(self.groundTruthPoses[firstFrameCnt])
        self.prevFrame = self.kittiLoader.getFrame(firstFrameCnt)

        if self.plotProgress:
            self.plotter.setupPlot()

        while True:
            if self.currFrameCnt == lastFrameCnt:
                break

            self.currFrame = self.kittiLoader.getFrame(self.currFrameCnt)

            if motionEstimation.lower() == '2d-2d' or isinstance(self.prevPrevFrame, type(None)):
                R, t = self.estimateMotion2dTo2d(useGroundTruthScale)
            else:
                R, t = self.estimateMotion3dTo2d()

            tUnit = t / np.linalg.norm(t)
            if not np.logical_and.reduce((np.abs(tUnit) < 0.2) | (np.abs(tUnit) > 0.9))[0] and tUnit[2, 0] < 0:
                self.estimatedPoses.append(self.estimatedPoses[-1])
            else:
                self.estimatedPoses.append(self.estimatedPoses[-1]
                                           @ np.linalg.inv(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))))

                self.prevFrameCnt = self.currFrameCnt
                self.prevPrevFrame = self.prevFrame
                self.prevFrame = self.currFrame

            self.currFrameCnt += 1

            if self.plotProgress:
                self.plotter.updatePlot(self.currFrame, self.currFrameKeypoints)

        if self.plotProgress:
            self.plotter.plotResult()

        return np.array(self.estimatedPoses), self.groundTruthPoses


    def getRelativeScale(self, R, t):
        matchedKeypoints1, matchedKeypoints2, matchedKeypoints3 = \
            self.findFeatureMatches3Frames(self.prevPrevFrame, self.prevFrame, self.currFrame)

        matchedKeypoints1 = cv.undistortPoints(matchedKeypoints1, self.K, np.array([]))
        matchedKeypoints2 = cv.undistortPoints(matchedKeypoints2, self.K, np.array([]))
        matchedKeypoints3 = cv.undistortPoints(matchedKeypoints3, self.K, np.array([]))

        points3d12 = cv.triangulatePoints(np.eye(3) @ self.estimatedPoses[-2][0:3, :],
                                          np.eye(3) @ self.estimatedPoses[-1][0:3, :],
                                          matchedKeypoints1,
                                          matchedKeypoints2)

        points3d12 = np.transpose(points3d12)
        points3d12[:, 0] /= points3d12[:, 3]
        points3d12[:, 1] /= points3d12[:, 3]
        points3d12[:, 2] /= points3d12[:, 3]
        points3d12 = points3d12[:, 0:3]

        points3d23 = cv.triangulatePoints(np.eye(3) @ self.estimatedPoses[-1][0:3, :],
                                          np.eye(3) @ np.hstack((R, t)),
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scales = np.linalg.norm(points3d12[i] - points3d12[j], axis=1) \
                     / np.linalg.norm(points3d23[i] - points3d23[j], axis=1)
            scales = scales[~np.isnan(scales) & (scales < 3) & (scales > 0.3)]

            return np.mean(scales)


    def estimateMotion2dTo2d(self, useGroundTruthScale):
        matchedKeypoints1, matchedKeypoints2 = self.findFeatureMatches2Frames(self.prevFrame, self.currFrame)
        self.currFrameKeypoints = matchedKeypoints2

        E, mask1 = cv.findEssentialMat(matchedKeypoints1, matchedKeypoints2, self.K,
                                       threshold=1, method=cv.RANSAC)
        _, R, t, mask2 = cv.recoverPose(E, matchedKeypoints1, matchedKeypoints2, self.K, mask=mask1)
        scaleCalculated = False
        if not useGroundTruthScale and not isinstance(self.prevPrevFrame, type(None)):
            scale = self.getRelativeScale(R, t)
            print(self.kittiLoader.getGroundTruthScale(self.prevFrameCnt, self.currFrameCnt), scale)
            scaleCalculated = not np.isnan(scale)
        if not scaleCalculated:
            scale = self.kittiLoader.getGroundTruthScale(self.prevFrameCnt, self.currFrameCnt)
        t = t * scale

        return R, t


    def estimateMotion3dTo2d(self):
        matchedKeypoints1, matchedKeypoints2, matchedKeypoints3 = \
            self.findFeatureMatches3Frames(self.prevPrevFrame, self.prevFrame, self.currFrame)
        self.currFrameKeypoints = matchedKeypoints3

        matchedKeypoints1 = cv.undistortPoints(matchedKeypoints1, self.K, np.array([]))
        matchedKeypoints2 = cv.undistortPoints(matchedKeypoints2, self.K, np.array([]))

        points3d = cv.triangulatePoints(np.eye(3) @ self.estimatedPoses[-2][0:3, :],
                                        np.eye(3) @ self.estimatedPoses[-1][0:3, :],
                                        matchedKeypoints1,
                                        matchedKeypoints2)

        points3d = np.transpose(points3d)
        points3d[:, 0] /= points3d[:, 3]
        points3d[:, 1] /= points3d[:, 3]
        points3d[:, 2] /= points3d[:, 3]
        points3d = points3d[:, 0:3]

        _, rvec, t, _ = cv.solvePnPRansac(points3d, matchedKeypoints3, self.K, np.array([]))
        R, _ = cv.Rodrigues(rvec)
        A = np.linalg.inv(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))) @ self.estimatedPoses[-1]

        return A[0:3, 0:3], A[0:3, 3].reshape((3, 1))
