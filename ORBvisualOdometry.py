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





# from visual_odometry import VisualOdometry
# from plotter import Plotter
#
# sequences = {
#     0: r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\00",
#     1: r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\01",
#     2: r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\02",
#     3: r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\03",
#     4: r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\04",
#     5: r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\05",
#     6: r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\06",
#     7: r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\07",
#     8: r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\08",
#     9: r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\09",
#     10: r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\10",
# }
#
#
# VO = VisualOdometry(sequences[5], True)
# estimatedPoses, groundTruthPoses = VO.run()
# Plotter.plotResult()
#
# exit(0)



from plotter import Plotter
from kitti_sequence_loader import KITTISequenceLoader
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def splitImages(numOfPartsWidth, numOfPartsHeight, *images):
    imageHeight = np.shape(images[0])[0]
    imageWidth = np.shape(images[0])[1]
    imagesVec = []
    imageShift = []

    for i in range(numOfPartsHeight):
        for j in range(numOfPartsWidth):
            imageShift.append((int(i*imageHeight/numOfPartsHeight),
                               int(j*imageWidth/numOfPartsWidth)))
            imagesVec.append(tuple([image[int(i*imageHeight/numOfPartsHeight):int((i+1)*imageHeight/numOfPartsHeight),
                                          int(j*imageWidth/numOfPartsWidth):int((j+1)*imageWidth/numOfPartsWidth)]
                                    for image in images]))

    return imagesVec, imageShift


loader = KITTISequenceLoader(r"D:\Fakultet\Drugi ciklus\Prva godina\Drugi semestar\Robotska vizija\Seminarski\KITTI Dataset\05")
K = loader.getIntrinsicCameraParameters()




orb = cv.ORB_create(nfeatures=500, scoreType=cv.ORB_FAST_SCORE)
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

groundTruthPoses = loader.getAllGroundTruthPoses()
estimatedPoses = []
plotter = Plotter(groundTruthPoses, estimatedPoses)
plotter.setupPlot()

length = len(groundTruthPoses)
cnt1 = 50
cnt2 = cnt1 + 1
cnt3 = cnt2 + 2
estimatedPoses.append(groundTruthPoses[cnt1])
while True:
    if cnt1 >= 80:
        break

    image1 = loader.getFrame(cnt1)
    image2 = loader.getFrame(cnt2)
    image3 = loader.getFrame(cnt3)

    imagesVec, imageShift = splitImages(3, 2, image1, image2, image3)

    matchedKeypoints1 = []
    matchedKeypoints2 = []
    matchedKeypoints3 = []

    for cnt, images in enumerate(imagesVec):
        keypoints1, descriptors1 = orb.detectAndCompute(images[0], None)
        keypoints2, descriptors2 = orb.detectAndCompute(images[1], None)
        keypoints3, descriptors3 = orb.detectAndCompute(images[2], None)

        if type(descriptors1) == type(None) or type(descriptors2) == type(None) or type(descriptors3) == type(None):
            continue

        matches12 = matcher.match(descriptors1, descriptors2)
        matches23 = matcher.match(descriptors2, descriptors3)
        matches13 = matcher.match(descriptors1, descriptors3)

        for match12 in matches12:
            for match23 in matches23:
                if match12.trainIdx == match23.queryIdx:
                    (x1, y1) = keypoints1[match12.queryIdx].pt
                    (x2, y2) = keypoints2[match12.trainIdx].pt
                    (x3, y3) = keypoints3[match23.trainIdx].pt

                    y1 += imageShift[cnt][0]
                    y2 += imageShift[cnt][0]
                    y3 += imageShift[cnt][0]
                    x1 += imageShift[cnt][1]
                    x2 += imageShift[cnt][1]
                    x3 += imageShift[cnt][1]

                    matchedKeypoints1.append((x1, y1))
                    matchedKeypoints2.append((x2, y2))
                    matchedKeypoints3.append((x3, y3))
                    break

                #     for match13 in matches13:
                #         if match12.queryIdx == match13.queryIdx:
                #         # if match12.queryIdx == match13.queryIdx and match23.trainIdx == match13.trainIdx:
                #             (x1, y1) = keypoints1[match12.queryIdx].pt
                #             (x2, y2) = keypoints2[match12.trainIdx].pt
                #             (x3, y3) = keypoints3[match23.trainIdx].pt
                #
                #             y1 += imageShift[cnt][0]
                #             y2 += imageShift[cnt][0]
                #             y3 += imageShift[cnt][0]
                #             x1 += imageShift[cnt][1]
                #             x2 += imageShift[cnt][1]
                #             x3 += imageShift[cnt][1]
                #
                #             matchedKeypoints1.append((x1, y1))
                #             matchedKeypoints2.append((x2, y2))
                #             matchedKeypoints3.append((x3, y3))
                #             break


    matchedKeypoints1 = np.array(matchedKeypoints1)
    matchedKeypoints2 = np.array(matchedKeypoints2)
    matchedKeypoints3 = np.array(matchedKeypoints3)

    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.imshow(cv.drawKeypoints(image1, cv.KeyPoint_convert(matchedKeypoints1), None, color=(0, 255, 0), flags=0),
    #            cmap='gray')
    # plt.subplot(3, 1, 2)
    # plt.imshow(cv.drawKeypoints(image2, cv.KeyPoint_convert(matchedKeypoints2), None, color=(0, 255, 0), flags=0),
    #            cmap='gray')
    # plt.subplot(3, 1, 3)
    # plt.imshow(cv.drawKeypoints(image3, cv.KeyPoint_convert(matchedKeypoints3), None, color=(0, 255, 0), flags=0),
    #            cmap='gray')
    # plt.show()


    E, mask1 = cv.findEssentialMat(matchedKeypoints1, matchedKeypoints2, K, threshold=1, method=cv.RANSAC)
    _, R, t, mask2 = cv.recoverPose(E, matchedKeypoints1, matchedKeypoints2, K, mask=mask1)
    scale = loader.getGroundTruthScale(cnt1, cnt2)
    t = t * scale

    if cnt1 == 50:
        estimatedPoses.append(estimatedPoses[-1]
                              @ np.linalg.inv(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))))


    # matchedKeypoints1 = matchedKeypoints1[mask2[:, 0] == 1]
    # matchedKeypoints2 = matchedKeypoints2[mask2[:, 0] == 1]
    # matchedKeypoints3 = matchedKeypoints3[mask2[:, 0] == 1]

    # pointsHom = cv.triangulatePoints(K @ groundTruthPoses[cnt1][0:3, :], K @ groundTruthPoses[cnt2][0:3, :],
    #                                  np.transpose(matchedKeypoints1), np.transpose(matchedKeypoints2))
    pointsHom = cv.triangulatePoints(K @ estimatedPoses[-2][0:3, :], K @ estimatedPoses[-1][0:3, :],
                                     np.transpose(matchedKeypoints1), np.transpose(matchedKeypoints2))


    pointsHom = np.transpose(pointsHom)
    pointsHom[:, 0] /= pointsHom[:, 3]
    pointsHom[:, 1] /= pointsHom[:, 3]
    pointsHom[:, 2] /= pointsHom[:, 3]
    points3D = pointsHom[:, 0:3]

    _, rvec, t, _ = cv.solvePnPRansac(points3D, matchedKeypoints3, K, np.array([]))

    R, _ = cv.Rodrigues(rvec)

    # tmp = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
    # relativeTransformation = tmp @ np.linalg.inv(estimatedPoses[-1])
    # print(relativeTransformation)
    relativeTranslation = np.transpose(t)[0] - estimatedPoses[-1][0:3, 3]
    # print(estimatedPoses[-1][0:3, 0:3] @ relativeTransformation[0:3, 3].reshape((3, 1)))
    # print(relativeTranslation)
    # if relativeTranslation[2] > 0 and np.abs(relativeTranslation[0]) < 0.1:
    #     estimatedPoses.append(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1]))))
    #     cnt1 = cnt2
    #     cnt2 = cnt3

    estimatedPoses.append(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1]))))
    cnt1 = cnt2
    cnt2 = cnt3

    # print(np.linalg.norm(np.transpose(t)[0] - groundTruthPoses[i + 2][0:3, 3]))
    # if np.linalg.norm(np.transpose(t)[0] - groundTruthPoses[i + 2][0:3, 3]) < 5:
    #     estimatedPoses.append(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1]))))
    # else:
    #     estimatedPoses.append(groundTruthPoses[i + 2])

    # print(groundTruthPoses[i + 2][0:3, 3])
    # print(np.transpose(t)[0])
    # print("\n")

    cnt3 += 1
    plotter.updatePlot(image3, matchedKeypoints3)


plotter.plotResult()