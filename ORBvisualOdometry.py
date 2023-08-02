# import numpy as np
# import cv2 as cv
#
# from kitti_sequence_loader import KITTISequenceLoader
# from plotter import Plotter
#
# from timeit import default_timer as timer
#
# loader = KITTISequenceLoader(r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\09")
# plotter = None
#
# orb = cv.ORB_create(500, scoreType=cv.ORB_FAST_SCORE)
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
#
#
# def splitImages(numOfPartsWidth, numOfPartsHeight, *images):
#     imageHeight = np.shape(images[0])[0]
#     imageWidth = np.shape(images[0])[1]
#     imagesVec = []
#     imageShift = []
#
#     for i in range(numOfPartsHeight):
#         for j in range(numOfPartsWidth):
#             imageShift.append((int(i*imageHeight/numOfPartsHeight),
#                                int(j*imageWidth/numOfPartsWidth)))
#             imagesVec.append(tuple([image[int(i*imageHeight/numOfPartsHeight):int((i+1)*imageHeight/numOfPartsHeight),
#                                           int(j*imageWidth/numOfPartsWidth):int((j+1)*imageWidth/numOfPartsWidth)]
#                                     for image in images]))
#
#     return imagesVec, imageShift
#
#
# def detectComputeAndMatchORB(image1, image2):
#     matchedKeypoints1 = []
#     matchedKeypoints2 = []
#
#     imagesVec, imageShift = splitImages(3, 2, image1, image2)
#
#     for cnt, images in enumerate(imagesVec):
#         keypoints1, descriptors1 = orb.detectAndCompute(images[0], None)
#         keypoints2, descriptors2 = orb.detectAndCompute(images[1], None)
#
#         if type(descriptors1) == type(None) or type(descriptors2) == type(None):
#             continue
#
#         matches = bf.match(descriptors1, descriptors2)
#
#         for match in matches:
#             (x1, y1) = keypoints1[match.queryIdx].pt
#             (x2, y2) = keypoints2[match.trainIdx].pt
#
#             y1 += imageShift[cnt][0]
#             y2 += imageShift[cnt][0]
#             x1 += imageShift[cnt][1]
#             x2 += imageShift[cnt][1]
#
#             matchedKeypoints1.append((x1, y1))
#             matchedKeypoints2.append((x2, y2))
#
#     return np.array(matchedKeypoints1), np.array(matchedKeypoints2)
#
#
#
# def runVO(firstFrameCnt = 0, lastFrameCnt = None):
#     estimatedPoses = [np.eye(4)]
#     K = loader.getIntrinsicCameraParameters()
#     groundTruthPoses = loader.getAllGroundTruthPoses()
#
#     if lastFrameCnt is None:
#         lastFrameCnt = loader.getNumberOfFrames()
#
#     prevFrameCnt = firstFrameCnt
#     currFrameCnt = prevFrameCnt + 1
#
#     estimatedPoses[0] = groundTruthPoses[prevFrameCnt]
#
#     global plotter
#     plotter = Plotter(groundTruthPoses, estimatedPoses)
#     plotter.setupPlot()
#
#     firstIter = True
#     while True:
#         # start = timer()
#         if currFrameCnt == lastFrameCnt:
#             break
#
#         image1 = loader.getFrame(prevFrameCnt)
#         image2 = loader.getFrame(currFrameCnt)
#
#         matchedKeypoints1, matchedKeypoints2 = detectComputeAndMatchORB(image1, image2)
#         # print(np.shape(matchedKeypoints1))
#         if firstIter:
#             plotter.updatePlot(image1, matchedKeypoints1)
#             firstIter = False
#
#         E, mask1 = cv.findEssentialMat(matchedKeypoints1, matchedKeypoints2, K, threshold=1, method=cv.RANSAC)
#         _, R, t, mask2 = cv.recoverPose(E, matchedKeypoints1, matchedKeypoints2, K, mask=mask1)
#         scale = loader.getGroundTruthScale(prevFrameCnt, currFrameCnt)
#         t = t * scale
#
#         if scale < 0.3:
#             estimatedPoses.append(estimatedPoses[-1])
#             currFrameCnt += 1
#         else:
#             estimatedPoses.append(estimatedPoses[-1]
#                                   @ np.linalg.inv(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))))
#             prevFrameCnt = currFrameCnt
#             currFrameCnt += 1
#
#         plotter.updatePlot(image2, matchedKeypoints2)
#         # print(timer()-start)
#
#     return np.array(estimatedPoses)

from visual_odometry import VisualOdometry

VO = VisualOdometry(r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\09",
                    True)

estimatedPoses, groundTruthPoses = VO.run()
# estimatedPoses = runVO()
# groundTruthPoses = loader.getAllGroundTruthPoses()
# plotter.plotResult()
