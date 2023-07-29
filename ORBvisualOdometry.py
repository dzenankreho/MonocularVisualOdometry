import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


orb = cv.ORB_create(2000)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

FLANN_INDEX_LSH = 6
index_params= dict(algorithm=FLANN_INDEX_LSH,
                   table_number=12, # 12 6
                   key_size=20, # 20 12
                   multi_probe_level=2) #2 1
flann = cv.FlannBasedMatcher(index_params)


def getIntrinsicParamFromFile(filename):
    with open(filename, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]

    return K


def getGroundTruthFromFile(filename):
    cnt = 0
    groundTruthPoses = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            # if cnt == 500:
            #     break
            pose = np.fromstring(line, dtype=np.float64, sep=' ')
            pose = pose.reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            groundTruthPoses.append(pose)
            cnt += 1

    return np.array(groundTruthPoses)


def detectComputeAndMatchORB(image1, image2):
    imageHeight = np.shape(image1)[0]
    imageWidth = np.shape(image1)[1]

    image11 = image1[0:int(imageHeight/2), 0:int(imageWidth/2)]
    image21 = image2[0:int(imageHeight/2), 0:int(imageWidth/2)]
    image12 = image1[0:int(imageHeight/2), int(imageWidth/2):imageWidth]
    image22 = image2[0:int(imageHeight/2), int(imageWidth/2):imageWidth]
    image13 = image1[int(imageHeight/2):imageHeight, 0:int(imageWidth/2)]
    image23 = image2[int(imageHeight/2):imageHeight, 0:int(imageWidth/2)]
    image14 = image1[int(imageHeight/2):imageHeight, int(imageWidth/2):imageWidth]
    image24 = image2[int(imageHeight/2):imageHeight, int(imageWidth/2):imageWidth]


    matchedKeypoints1 = []
    matchedKeypoints2 = []


    for cnt, images in enumerate([(image11, image21), (image12, image22), (image13, image23), (image14, image24)]):
        keypoints1, descriptors1 = orb.detectAndCompute(images[0], None)
        keypoints2, descriptors2 = orb.detectAndCompute(images[1], None)

        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        for match in matches:
            (x1, y1) = keypoints1[match.queryIdx].pt
            (x2, y2) = keypoints2[match.trainIdx].pt

            if cnt == 1:
                x1 += int(imageWidth / 2)
                x2 += int(imageWidth / 2)
            elif cnt == 2:
                y1 += int(imageHeight / 2)
                y2 += int(imageHeight / 2)
            elif cnt == 3:
                x1 += int(imageWidth / 2)
                x2 += int(imageWidth / 2)
                y1 += int(imageHeight / 2)
                y2 += int(imageHeight / 2)


            matchedKeypoints1.append((x1, y1))
            matchedKeypoints2.append((x2, y2))


    return np.array(matchedKeypoints1), np.array(matchedKeypoints2)

# def detectComputeAndMatchORB(image1, image2):
#     # imageHeight = np.shape(image1)[0]
#     # imageWidth = np.shape(image1)[1]
#     #
#     # image11 = image1[0:int(imageHeight/2), 0:int(imageWidth/2)]
#     # image21 = image2[0:int(imageHeight/2), 0:int(imageWidth/2)]
#     # image12 = image1[0:int(imageHeight/2), int(imageWidth/2)+1:imageWidth-1]
#     # image22 = image2[0:int(imageHeight/2), int(imageWidth/2)+1:imageWidth-1]
#     # image13 = image1[int(imageHeight/2)+1:imageHeight-1, 0:int(imageWidth/2)]
#     # image23 = image2[int(imageHeight/2)+1:imageHeight-1, 0:int(imageWidth/2)]
#     # image14 = image1[int(imageHeight/2)+1:imageHeight-1, int(imageWidth/2)+1:imageWidth-1]
#     # image24 = image2[int(imageHeight/2)+1:imageHeight-1, int(imageWidth/2)+1:imageWidth-1]
#     #
#     # keypoints1 = []
#     # keypoints2 = []
#     # descriptors1 = []
#     # descriptors2 = []
#
#     keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
#     keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
#
#     # for images in [(image11, image21), (image12, image22), (image13, image23), (image14, image24)]:
#     #     kp1, des1 = orb.detectAndCompute(images[0], None)
#     #     kp2, des2 = orb.detectAndCompute(images[1], None)
#     #     keypoints1.extend(kp1)
#     #     keypoints2.extend(kp2)
#     #     descriptors1.extend(des1)
#     #     descriptors2.extend(des2)
#     #
#     # keypoints1 = np.array(keypoints1)
#     # keypoints2 = np.array(keypoints2)
#     # descriptors1 = np.array(descriptors1)
#     # descriptors2 = np.array(descriptors2)
#
#
#     matches = bf.match(descriptors1, descriptors2)
#     # matches = bf.knnMatch(descriptors1, descriptors2, 2)
#     # matches = flann.match(descriptors1, descriptors2)
#     matches = sorted(matches, key=lambda x: x.distance)
#
#     matchedKeypoints1 = []
#     matchedKeypoints2 = []
#
#     for match in matches:
#         (x1, y1) = keypoints1[match.queryIdx].pt
#         (x2, y2) = keypoints2[match.trainIdx].pt
#
#         matchedKeypoints1.append((x1, y1))
#         matchedKeypoints2.append((x2, y2))
#
#     return np.array(matchedKeypoints1), np.array(matchedKeypoints2)


def runVO():
    gd = getGroundTruthFromFile(r'KITTI03\03.txt')
    scale = np.linalg.norm(np.diff(gd[0:len(gd):1], axis=0)[:, 0:3, 3], axis=1)

    estimatedPoses = [np.eye(4)]

    K = getIntrinsicParamFromFile(r'KITTI03\calib.txt')

    machedKeypoints = []
    points3D = []

    cnt = 0
    cnt2 = 0
    while True:
        if not os.path.exists("KITTI03\\image_0\\" + str(cnt + 1).zfill(6) + ".png"):
            break

        image1 = cv.imread("KITTI03\\image_0\\" + str(cnt).zfill(6) + ".png", cv.IMREAD_GRAYSCALE)
        image2 = cv.imread("KITTI03\\image_0\\" + str(cnt + 1).zfill(6) + ".png", cv.IMREAD_GRAYSCALE)

        matchedKeypoints1, matchedKeypoints2 = detectComputeAndMatchORB(image1, image2)
        machedKeypoints.append((matchedKeypoints1, matchedKeypoints2))


        # lk_params = dict(winSize=(15, 15),
        #                  maxLevel=2,
        #                  criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
        #                            10, 0.03))
        #
        # p0 = np.array([[kp] for kp in matchedKeypoints1], dtype=np.float32)
        # p1, st, err = cv.calcOpticalFlowPyrLK(image1, image2, p0,
        #                                       None, **lk_params)
        # good_new = p0[st == 1]
        # good_old = p1[st == 1]
        #
        # # matchedKeypoints1 = cv.KeyPoint_convert(good_new)
        # # matchedKeypoints2 = cv.KeyPoint_convert(good_old)
        # matchedKeypoints1 = good_old
        # matchedKeypoints2 = good_new


        E, mask1 = cv.findEssentialMat(matchedKeypoints1, matchedKeypoints2, K, threshold=1, method=cv.RANSAC)
        _, R, t, mask2 = cv.recoverPose(E, matchedKeypoints1, matchedKeypoints2, K, mask=mask1)
        t = t * scale[cnt2]

        #
        # pts3D = cv.triangulatePoints(K @ estimatedPoses[-1][0:3, :],
        #                              K @ np.hstack((R, t)),
        #                              np.transpose(matchedKeypoints1),
        #                              np.transpose(matchedKeypoints2))
        #
        # pts3D = np.transpose(pts3D)
        # pts3D[:, 0] /= pts3D[:, 3]
        # pts3D[:, 1] /= pts3D[:, 3]
        # pts3D[:, 2] /= pts3D[:, 3]
        # pts3D[:, 3] /= pts3D[:, 3]
        # pts3D = pts3D[:, 0:3]
        # points3D.append(pts3D)
        #

        # if cnt > 0:
        #     machedKeypointIntersections = np.array(list(set((tuple(i) for i in machedKeypoints[-2][1]))
        #                                                 .intersection(set((tuple(i) for i in machedKeypoints[-1][0])))))
        #     machedKeypointIntersectionIndices = []
        #     for machedKeypoint in machedKeypointIntersections:
        #         machedKeypointIntersectionIndices.append(
        #             [np.argwhere(np.logical_and(machedKeypoints[-2][1][:, 0] == machedKeypoint[0],
        #                                         machedKeypoints[-2][1][:, 1] == machedKeypoint[1]))[0][0],
        #              np.argwhere(np.logical_and(machedKeypoints[-1][0][:, 0] == machedKeypoint[0],
        #                                         machedKeypoints[-1][0][:, 1] == machedKeypoint[1]))[0][0]]
        #         )
        #     machedKeypointIntersectionIndices = np.array(machedKeypointIntersectionIndices)
        #
        #     print(np.mean(np.linalg.norm(np.diff(points3D[cnt2 - 1][machedKeypointIntersectionIndices[:, 0]], axis=0), axis=1)
        #                   / np.linalg.norm(np.diff(points3D[cnt2][machedKeypointIntersectionIndices[:, 1]], axis=0),
        #                                    axis=1)), np.median(
        #         np.linalg.norm(np.diff(points3D[cnt2 - 1][machedKeypointIntersectionIndices[:, 0]], axis=0), axis=1)
        #         / np.linalg.norm(np.diff(points3D[cnt2][machedKeypointIntersectionIndices[:, 1]], axis=0),
        #                          axis=1)),
        #           np.linalg.norm(points3D[cnt2 - 1][machedKeypointIntersectionIndices[0, 0]]) /
        #           np.linalg.norm(points3D[cnt2][machedKeypointIntersectionIndices[0, 1]]))
        #
        # if cnt == 2:
        #     # print(len(points3D))
        #     # print(machedKeypoints[-1][0][machedKeypointIntersectionIndices[0][1]])
        #     # print(points3D[0][machedKeypointIntersectionIndices[:, 0]])
        #     # print(points3D[1][machedKeypointIntersectionIndices[:, 1]])
        #
        #     # print(points3D[0][machedKeypointIntersectionIndices[:, 0]])
        #     print(
        #         np.median(np.linalg.norm(np.diff(points3D[0][machedKeypointIntersectionIndices[:, 0]], axis=0), axis=1)
        #                   / np.linalg.norm(np.diff(points3D[1][machedKeypointIntersectionIndices[:, 0]], axis=0),
        #                                    axis=1)))
        #     # print(np.linalg.norm(points3D[0][machedKeypointIntersectionIndices[:, 0]][0]
        #     #       - points3D[0][machedKeypointIntersectionIndices[:, 0]][1])
        #     #       / np.linalg.norm(points3D[1][machedKeypointIntersectionIndices[:, 0]][0]
        #     #       - points3D[1][machedKeypointIntersectionIndices[:, 0]][1]))
        #     # print(points3D[0][machedKeypointIntersectionIndices[:, 0]] - points3D[1][machedKeypointIntersectionIndices[:, 1]])
        #
        #     # print(machedKeypointsSetIntersection)
        #     # print(np.array(list(machedKeypointsSetIntersection))[0])
        #     # print(machedKeypoints[-2][1][:, 0] == np.array(list(machedKeypointsSetIntersection))[0][0])
        #     # print(machedKeypoints[-2][1][:, 1] == np.array(list(machedKeypointsSetIntersection))[0][1])
        #     # print(np.argwhere(np.logical_and(
        #     #     machedKeypoints[-2][1][:, 0] == np.array(list(machedKeypointsSetIntersection))[0][0],
        #     #     machedKeypoints[-2][1][:, 1] == np.array(list(machedKeypointsSetIntersection))[0][1]))[0, 0])
        #     # print(np.argwhere(np.logical_and(
        #     #     machedKeypoints[-1][0][:, 0] == np.array(list(machedKeypointsSetIntersection))[0][0],
        #     #     machedKeypoints[-1][0][:, 1] == np.array(list(machedKeypointsSetIntersection))[0][1]))[0, 0])
        #
        #     # for keypoint in machedKeypoints[-2][1]:
        #     #     print(keypoint)
        #     # print("---------\n")
        #     # for keypoint in machedKeypoints[-1][0]:
        #     #     print(keypoint)
        #     # _, ind1x, ind2x = np.intersect1d(machedKeypoints[-2][1][:, 0],
        #     #                                  machedKeypoints[-1][0][:, 0],
        #     #                                  return_indices=True)
        #     # _, ind1y, ind2y = np.intersect1d(machedKeypoints[-2][1][:, 1],
        #     #                                  machedKeypoints[-1][0][:, 1],
        #     #                                  return_indices=True)
        #     # print("---------\n")
        #     # print(machedKeypoints[-2][1][ind1x])
        #     # print(ind1x)
        #     # print(ind2x)
        #     # print(ind1y)
        #     # print(ind2y)
        #     # print(ind1x == ind1y)
        #
        #     break

        estimatedPoses.append(estimatedPoses[-1]
                              @ np.linalg.inv(np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))))

        # if estimatedPoses[-1][2, 3] > 135:
        #     print(np.shape(matchedKeypoints1))

        cnt += 1
        cnt2 += 1

    return np.array(estimatedPoses)


groundTruthPoses = getGroundTruthFromFile(r'KITTI03\03.txt')
estimatedPoses = runVO()

x1 = []
z1 = []
i = 0
tmp = np.linalg.norm(np.diff(groundTruthPoses, axis=0)[:, 0:3, 3], axis=1)
for pose in estimatedPoses:
    if i == 0:
        x1.append(pose[0][3])
        z1.append(pose[2][3])
    else:
        x1.append(pose[0][3] * tmp[i - 1])
        z1.append(pose[2][3] * tmp[i - 1])
    i += 1

plt.figure()
plt.plot(groundTruthPoses[:, 0, 3], groundTruthPoses[:, 2, 3], 'r',
         # x1, z1, 'b')
         estimatedPoses[:, 0, 3], estimatedPoses[:, 2, 3], 'b')

# print(np.linalg.norm(np.diff(groundTruthPoses, axis=0)[:, 0:3, 3], axis=1))
# print(np.linalg.norm(np.diff(estimatedPoses, axis=0)[:, 0:3, 3], axis=1))

plt.show()

# tmp = np.linalg.norm(np.diff(groundTruthPoses, axis=0)[:, 0:3, 3], axis=1)
