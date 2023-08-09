from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv


class Plotter:
    """Plotter for showing progress and results of visual odometry
    """

    def __init__(self, groundTruthPoses, estimatedPoses):
        """Initializes the instance with lists of ground truth and estimated poses

        Args:
            groundTruthPoses (list): List of all ground truth poses
            estimatedPoses (list): List containing only the initial pose
        """
        self.groundTruthPoses = groundTruthPoses
        self.estimatedPoses = estimatedPoses
        self.figure = None
        self.closed = False


    def onClose(self, event):
        self.closed = True


    def setupPlot(self):
        """Creates a figure for plotting progress (must be called before updatePlot)
        """
        self.figure = plt.figure()
        plt.pause(0.2)
        self.figure.canvas.mpl_connect('close_event', self.onClose)


    def updatePlot(self, frame, keypoints):
        """Updates the progress figure with current frame and matched features

        Args:
            frame (ndarray): Current frame in grayscale
            keypoints (ndarray): Locations of matched features in pixels
        """
        if self.closed:
            return

        plt.figure(self.figure.number)
        plt.clf()
        plt.subplot(2, 2, (1, 3))
        plt.plot(self.groundTruthPoses[:, 0, 3], self.groundTruthPoses[:, 2, 3], 'r',
                 np.array(self.estimatedPoses)[:, 0, 3], np.array(self.estimatedPoses)[:, 2, 3], 'b')
        plt.title(r"Trajectory", wrap=True)
        plt.legend([r"Ground truth", r"Estimated"])
        plt.subplot(2, 2, 2)
        plt.imshow(frame, cmap='gray')
        plt.title(r"Current image", wrap=True)
        plt.subplot(2, 2, 4)
        plt.imshow(cv.drawKeypoints(frame, cv.KeyPoint_convert(keypoints), None, color=(0, 255, 0), flags=0),
                   cmap='gray')
        plt.title(r"Current image with detected ORB features", wrap=True)
        plt.show(block=False)
        plt.pause(0.1)


    def plotResult(self):
        """ Plots the final trajectory
        """
        plt.close(self.figure)
        plt.figure()
        plt.plot(self.groundTruthPoses[:, 0, 3], self.groundTruthPoses[:, 2, 3], 'r',
                 np.array(self.estimatedPoses)[:, 0, 3], np.array(self.estimatedPoses)[:, 2, 3], 'b')
        plt.title(r"Trajectory", wrap=True)
        plt.legend([r"Ground truth", r"Estimated"])
        plt.show()
