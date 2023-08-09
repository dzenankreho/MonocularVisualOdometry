from KITTISequenceLoader import KITTISequenceLoader
from Plotter import Plotter
import numpy as np
import cv2 as cv
import warnings
from timeit import default_timer as timer


class VisualOdometry:
    """Monocular visual odometry using ORB features
    """

    def __init__(self, sequenceLocation, numOfFeatures=500, frameSplitParts=(3, 2),
                 matcher='bf', plotProgress=False):
        """Initializes the instance with the algorithm configurations and preferences

        Args:
            sequenceLocation (string): Path to the folder of the KITTI sequence
            numOfFeatures (int): Number of features to be detected per subimage
            frameSplitParts (tuple): Number of parts into which to split the image in width and height
            matcher (string): Matcher to be used ('bf' for brute-force or 'flann' for FLANN)
            plotProgress (bool): Indicates whether to plot the progress
        """
        self.plotProgress = plotProgress
        self.kittiLoader = KITTISequenceLoader(sequenceLocation)
        self.groundTruthPoses = self.kittiLoader.getAllGroundTruthPoses()
        self.K = self.kittiLoader.getIntrinsicCameraParameters()
        self.estimatedPoses = []

        self.orb = cv.ORB_create(nfeatures=numOfFeatures,
                                 scoreType=cv.ORB_FAST_SCORE)

        if matcher.lower() == 'bf':
            self.matcher = cv.BFMatcher(normType=cv.NORM_HAMMING,
                                        crossCheck=True)
        elif matcher.lower() == 'flann':
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
            search_params = dict(checks=50)
            self.matcher = cv.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError("Invalid matcher select variable")

        self.numOfPartsWidth = frameSplitParts[0]
        self.numOfPartsHeight = frameSplitParts[1]

        self.prevFrameCnt = None
        self.currFrameCnt = None
        self.prevPrevFrame = None
        self.prevFrame = None
        self.currFrame = None
        self.currFrameKeypoints = None

        if plotProgress:
            self.plotter = Plotter(self.groundTruthPoses, self.estimatedPoses)


    def splitImages(self, *images):
        """Splits the given images into parts using the defined grid

        Args:
            *images: Images to be split into parts

        Returns: An array containing the subimages for each image, and an array containing the location of the subimages in original image
        """
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
        """Detects and matches features between two images

        Args:
            image1 (ndarray): First image
            image2 (ndarray): Second image

        Returns:
            Arrays containing the locations of the matched features in each image
        """
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
        """Detects and matches features between three images

        Args:
            image1 (ndarray): First image
            image2 (ndarray): Second image
            image3 (ndarray): Third image

        Returns:
            Arrays containing the locations of the matched features in each image
        """
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


    def getRelativeScale(self, R, t):
        """Computes the relative scaling factor for the newly computed pose

        Args:
            R (ndarray): Relative rotation matrix between the previous and current poses
            t (ndarray): Relative translation vector between the previous and current poses

        Returns:
            Relative scaling factor
        """
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
        """Computes the relative rotation matrix and translation vector from 2D-2D correspondences

        Args:
            useGroundTruthScale (bool): Indicates whether to calculate or use the ground truth relative scale

        Returns:
            Relative rotation matrix and the relative translation vector
        """
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
        """Computes the relative rotation matrix and translation vector from 3D-2D correspondences

        Returns:
            Relative rotation matrix and the relative translation vector
        """
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


    def run(self, firstFrameCnt=0, lastFrameCnt=None, useGroundTruthScale=True, motionEstimation='2d-2d'):
        """Starts the monocular visual odometry

        Args:
            firstFrameCnt (int): Number of the frame from which to start the visual odometry
            lastFrameCnt (int): Number of the frame on which to end the visual odometry
            useGroundTruthScale (bool): Indicates whether to use relative ground truth scale when estimating motion from 2D-2D correspondences
            motionEstimation (string): Indicates what motion estimation method to use ('2d-2d' for 2D-2D or '3d-2d' for 3D-2D)

        Returns:
            Estimated trajectory (poses), ground truth trajectory (poses) and the average processing time of a frame
        """
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

        avgFrameProcessTime = 0

        while True:
            if self.currFrameCnt == lastFrameCnt:
                break

            start = timer()

            self.currFrame = self.kittiLoader.getFrame(self.currFrameCnt)

            if motionEstimation.lower() == '2d-2d' or isinstance(self.prevPrevFrame, type(None)):
                R, t = self.estimateMotion2dTo2d(useGroundTruthScale)
            else:
                R, t = self.estimateMotion3dTo2d()

            tUnit = t / np.linalg.norm(t)
            if not np.logical_and.reduce((np.abs(tUnit) < 0.2) | (np.abs(tUnit) > 0.9))[0]:
                self.estimatedPoses.append(self.estimatedPoses[-1])
            else:
                self.estimatedPoses.append(self.estimatedPoses[-1]
                                           @ np.linalg.inv(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))))

                self.prevFrameCnt = self.currFrameCnt
                self.prevPrevFrame = self.prevFrame
                self.prevFrame = self.currFrame

            self.currFrameCnt += 1

            avgFrameProcessTime += timer() - start

            if self.plotProgress:
                self.plotter.updatePlot(self.currFrame, self.currFrameKeypoints)

        if self.plotProgress:
            self.plotter.plotResult()

        avgFrameProcessTime /= (lastFrameCnt - firstFrameCnt)

        return np.array(self.estimatedPoses), self.groundTruthPoses[firstFrameCnt:lastFrameCnt], avgFrameProcessTime
