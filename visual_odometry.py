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
            keypoints3, descriptors3 = self.orb.detectAndCompute(images[3], None)

            if isinstance(descriptors1, type(None)) or isinstance(descriptors2, type(None))\
                    or isinstance(descriptors3, type(None)):
                continue

            matches12 = self.matcher.match(descriptors1, descriptors2)
            matches23 = self.matcher.match(descriptors2, descriptors3)

            for match12 in matches12:
                for match23 in matches23:
                    if match12.trainIdx == match23.queryIdx:
                        matchedKeypoints1.append(np.array(keypoints1[match12.queryIdx].pt) + imageShift[cnt])
                        matchedKeypoints2.append(keypoints2[match12.trainIdx].pt + imageShift[cnt])
                        matchedKeypoints3.append(keypoints3[match23.trainIdx].pt + imageShift[cnt])
                        break

        return np.array(matchedKeypoints1), np.array(matchedKeypoints2), np.array(matchedKeypoints3)


    def calculateEssentialMatrix(self, points1, points2):
        numOfPoints = np.shape(points1)[0]

        Q = np.hstack(((points2[:, 0] * points1[:, 0]).reshape((numOfPoints, 1)),
                       (points2[:, 0] * points1[:, 1]).reshape((numOfPoints, 1)),
                       (points2[:, 0] * np.ones((1, np.shape(points1)[0]))).reshape((numOfPoints, 1)),
                       (points2[:, 1] * points1[:, 0]).reshape((numOfPoints, 1)),
                       (points2[:, 1] * points1[:, 1]).reshape((numOfPoints, 1)),
                       (points2[:, 1] * np.ones((1, np.shape(points1)[0]))).reshape((numOfPoints, 1)),
                       (points1[:, 0]).reshape((numOfPoints, 1)),
                       (points1[:, 1]).reshape((numOfPoints, 1)),
                       (np.ones((1, np.shape(points1)[0]))).reshape((numOfPoints, 1))))

        _, _, Ev = np.linalg.svd(Q, full_matrices=True)
        E = Ev[-1]
        return E.reshape((3, 3))


    def run(self, firstFrameCnt=0, lastFrameCnt=None):
        K = self.kittiLoader.getIntrinsicCameraParameters()

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

            matchedKeypoints1, matchedKeypoints2 = self.findFeatureMatches2Frames(image1, image2)

            if self.plotProgress and firstIter:
                self.plotter.updatePlot(image1, matchedKeypoints1)
                firstIter = False

            E, mask1 = cv.findEssentialMat(matchedKeypoints1, matchedKeypoints2, K, threshold=1, method=cv.RANSAC)
            _, R, t, mask2 = cv.recoverPose(E, matchedKeypoints1, matchedKeypoints2, K, mask=mask1)
            scale = self.kittiLoader.getGroundTruthScale(prevFrameCnt, currFrameCnt)
            t = t * scale

            matchedKeypoints1 = matchedKeypoints1[mask2[:, 0] == 1]
            matchedKeypoints2 = matchedKeypoints2[mask2[:, 0] == 1]

            if scale < 0.3:
                self.estimatedPoses.append(self.estimatedPoses[-1])
            else:
                self.estimatedPoses.append(self.estimatedPoses[-1]
                                           @ np.linalg.inv(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))))

                # if False and currFrameCnt > 1:
                #     points4d = cv.triangulatePoints(K @ self.estimatedPoses[-2][0:3, :],
                #                                     K @ self.estimatedPoses[-1][0:3, :],
                #                                     np.transpose(matchedKeypoints1),
                #                                     np.transpose(matchedKeypoints2))
                #
                #     points4d = np.transpose(points4d)
                #     points4d[:, 0] /= points4d[:, 3]
                #     points4d[:, 1] /= points4d[:, 3]
                #     points4d[:, 2] /= points4d[:, 3]
                #     points3d = points4d[:, 0:3]
                #
                #     rvec, tvec = cv.solvePnPRefineLM(points3d, matchedKeypoints2, K, np.array([]),
                #                                      cv.Rodrigues(self.estimatedPoses[-1][0:3, 0:3])[0],
                #                                      self.estimatedPoses[-1][0:3, 3].reshape((3, 1)))
                #     self.estimatedPoses[-1] = np.vstack((np.hstack((cv.Rodrigues(rvec)[0], tvec)),
                #                                          np.array([0, 0, 0, 1])))

                prevFrameCnt = currFrameCnt

            currFrameCnt += 1

            if self.plotProgress:
                self.plotter.updatePlot(image2, matchedKeypoints2)
            # print(timer()-start)

        if self.plotProgress:
            self.plotter.plotResult()

        return np.array(self.estimatedPoses), self.groundTruthPoses
