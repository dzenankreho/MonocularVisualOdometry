from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv


class KITTIPlotter:
    def __init__(self, groundTruthPoses, estimatedPoses):
        self.groundTruthPoses = groundTruthPoses
        self.estimatedPoses = estimatedPoses
        self.figure = None
        self.closed = False


    def onClose(self, event):
        self.closed = True


    def setupPlot(self):
        self.figure = plt.figure()
        plt.pause(0.2)
        self.figure.canvas.mpl_connect('close_event', self.onClose)


    def updatePlot(self, frame, keypoints):
        if self.closed:
            return

        plt.figure(self.figure.number)
        plt.clf()
        plt.subplot(2, 2, (1, 3))
        plt.plot(self.groundTruthPoses[:, 0, 3], self.groundTruthPoses[:, 2, 3], 'r',
                 np.array(self.estimatedPoses)[:, 0, 3], np.array(self.estimatedPoses)[:, 2, 3], 'b')
        plt.subplot(2, 2, 2)
        plt.imshow(frame, cmap='gray')
        plt.subplot(2, 2, 4)
        plt.imshow(cv.drawKeypoints(frame, cv.KeyPoint_convert(keypoints), None, color=(0, 255, 0), flags=0),
                   cmap='gray')
        plt.show(block=False)
        plt.pause(0.1)


    def plotResult(self):
        plt.figure()
        plt.plot(self.groundTruthPoses[:, 0, 3], self.groundTruthPoses[:, 2, 3], 'r',
                 np.array(self.estimatedPoses)[:, 0, 3], np.array(self.estimatedPoses)[:, 2, 3], 'b')
        plt.show()
