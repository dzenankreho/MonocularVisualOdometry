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

        groundTruthPoses = []
        with open(self.sequenceLocation + "\\poses.txt", 'r') as f:
            for line in f.readlines():
                groundTruthPoses.append(np.vstack((np.fromstring(line, dtype=np.float64, sep=' ').reshape(3, 4),
                                                   [0, 0, 0, 1])))
        self.groundTruthPoses = np.array(groundTruthPoses)

        with open(self.sequenceLocation + "\\calib.txt", 'r') as f:
            self.K = np.reshape(np.fromstring(f.readline()[4:], dtype=np.float64, sep=' '), (3, 4))[0:3, 0:3]


    def getFrame(self, frameNumber):
        if frameNumber < 0 or frameNumber >= self.numberOfFrames:
            raise ValueError("Invalid frame number")

        frame = cv.imread(self.sequenceLocation + "\\image_0\\" + str(frameNumber).zfill(6) + ".png",
                          cv.IMREAD_GRAYSCALE)

        return frame


    def getIntrinsicCameraParameters(self):
        return self.K


    def getGroundTruthPose(self, frameNumber):
        if frameNumber < 0 or frameNumber >= self.numberOfFrames:
            raise ValueError("Invalid frame number")

        return self.groundTruthPoses[frameNumber]


    def getGroundTruthScale(self, prevFrameNumber, currFrameNumber):
        if prevFrameNumber < 0 or prevFrameNumber >= self.numberOfFrames \
                or currFrameNumber < 0 or currFrameNumber >= self.numberOfFrames:
            raise ValueError("Invalid frame number")

        return np.linalg.norm(self.groundTruthPoses[currFrameNumber][0:3, 3]
                              - self.groundTruthPoses[prevFrameNumber][0:3, 3])


    def getNumberOfFrames(self):
        return self.numberOfFrames


    def getAllGroundTruthPoses(self):
        return self.groundTruthPoses


    def playSequence(self):
        for i in range(self.numberOfFrames):
            cv.imshow("KITTI sequence", self.getFrame(i))
            key = cv.waitKey(2)
            if key == 27:
                cv.destroyAllWindows()
                break
