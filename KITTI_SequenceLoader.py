import cv2 as cv
import os
import numpy as np


class KITTISequenceLoader:
    def __init__(self, sequenceLocation):
        if not os.path.exists(sequenceLocation):
            raise OSError("Invalid sequence location")

        self.sequenceLocation = sequenceLocation
        self.numberOfFrames = len([file for file in os.listdir(self.sequenceLocation + "\\image_0")
                                   if file.endswith(".png")])


    def getFrame(self, frameNumber):
        if frameNumber < 0 or frameNumber >= self.numberOfFrames:
            raise ValueError("Invalid frame number")

        frame = cv.imread(self.sequenceLocation + "\\image_0\\" + str(frameNumber).zfill(6) + ".png",
                          cv.IMREAD_GRAYSCALE)

        return frame


    def getIntrinsicCameraParameters(self):
        with open(self.sequenceLocation + "\\calib.txt", 'r') as f:
            return np.reshape(np.fromstring(f.readline(), dtype=np.float64, sep=' '), (3, 4))[0:3, 0:3]


    def getGroundTruthPose(self, frameNumber):
        if frameNumber < 0 or frameNumber >= self.numberOfFrames:
            raise ValueError("Invalid frame number")

        with open(self.sequenceLocation + "\\poses.txt", 'r') as f:
            content = f.readlines()
            return np.vstack((np.fromstring(content[frameNumber], dtype=np.float64, sep=' ').reshape(3, 4),
                              [0, 0, 0, 1]))


    def getGroundTruthScale(self, prevFrameNumber, currFrameNumber):
        if prevFrameNumber < 0 or prevFrameNumber >= self.numberOfFrames \
                or currFrameNumber < 0 or currFrameNumber >= self.numberOfFrames:
            raise ValueError("Invalid frame number")

        return np.linalg.norm(self.getGroundTruthPose(currFrameNumber)[0:3, 3]
                              - self.getGroundTruthPose(prevFrameNumber)[0:3, 3])


    def getNumberOfFrames(self):
        return self.numberOfFrames


    def getAllGroundTruthPoses(self):
        return np.array([self.getGroundTruthPose(i) for i in range(self.numberOfFrames)])


    def playSequence(self):
        for i in range(self.numberOfFrames):
            cv.imshow("KITTI sequence", self.getFrame(i))
            key = cv.waitKey(2)
            if key == 27:
                cv.destroyAllWindows()
                break
