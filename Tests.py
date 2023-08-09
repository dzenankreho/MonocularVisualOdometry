from VisualOdometry import VisualOdometry
from matplotlib import pyplot as plt


kittiLocation = "D:\\Fakultet\\Drugi ciklus\\Prva godina\\Drugi semestar" \
                "\\Robotska vizija\\Seminarski\\KITTI Dataset\\"

sequences = dict()
for i in range(11):
    sequences[i] = kittiLocation + str(i).zfill(2)


# Testing different image grids using the FLANN matcher
VO = VisualOdometry(sequences[9], numOfFeatures=3000, frameSplitParts=(1, 1), matcher='flann')
estimatedPoses_1x1_3000_Flann, groundTruthPoses, avgFrameProcessTime_1x1_3000_Flann \
    = VO.run(firstFrameCnt=0, lastFrameCnt=1100)

VO = VisualOdometry(sequences[9], numOfFeatures=1500, frameSplitParts=(2, 1), matcher='flann')
estimatedPoses_2x1_1500_Flann, _, avgFrameProcessTime_2x1_1500_Flann \
    = VO.run(firstFrameCnt=0, lastFrameCnt=1100)
VO = VisualOdometry(sequences[9], numOfFeatures=750, frameSplitParts=(2, 2), matcher='flann')
estimatedPoses_2x2_750_Flann, _, avgFrameProcessTime_2x2_750_Flann \
    = VO.run(firstFrameCnt=0, lastFrameCnt=1100)
VO = VisualOdometry(sequences[9], numOfFeatures=500, frameSplitParts=(3, 2), matcher='flann')
estimatedPoses_3x2_500_Flann, _, avgFrameProcessTime_3x2_500_Flann \
    = VO.run(firstFrameCnt=0, lastFrameCnt=1100)

plt.figure()
plt.plot(groundTruthPoses[:, 0, 3], groundTruthPoses[:, 2, 3], 'r',
         estimatedPoses_1x1_3000_Flann[:, 0, 3], estimatedPoses_1x1_3000_Flann[:, 2, 3], 'b',
         estimatedPoses_2x1_1500_Flann[:, 0, 3], estimatedPoses_2x1_1500_Flann[:, 2, 3], 'g',
         estimatedPoses_2x2_750_Flann[:, 0, 3], estimatedPoses_2x2_750_Flann[:, 2, 3], 'm',
         estimatedPoses_3x2_500_Flann[:, 0, 3], estimatedPoses_3x2_500_Flann[:, 2, 3], 'y')
plt.title(r"Trajectories with FLANN matcher", wrap=True)
plt.legend([r"Ground truth", r"1x1 3000", r"2x1 1500", r"2x2 750", r"3x2 500"])


# Testing different image grids using the Brute-Force matcher
VO = VisualOdometry(sequences[9], numOfFeatures=3000, frameSplitParts=(1, 1), matcher='bf')
estimatedPoses_1x1_3000_BF, groundTruthPoses, avgFrameProcessTime_1x1_3000_BF \
    = VO.run(firstFrameCnt=0, lastFrameCnt=1100)
VO = VisualOdometry(sequences[9], numOfFeatures=1500, frameSplitParts=(2, 1), matcher='bf')
estimatedPoses_2x1_1500_BF, _, avgFrameProcessTime_2x1_1500_BF \
    = VO.run(firstFrameCnt=0, lastFrameCnt=1100)
VO = VisualOdometry(sequences[9], numOfFeatures=750, frameSplitParts=(2, 2), matcher='bf')
estimatedPoses_2x2_750_BF, _, avgFrameProcessTime_2x2_750_BF \
    = VO.run(firstFrameCnt=0, lastFrameCnt=1100)
VO = VisualOdometry(sequences[9], numOfFeatures=500, frameSplitParts=(3, 2), matcher='bf')
estimatedPoses_3x2_500_BF, _, avgFrameProcessTime_3x2_500_BF \
    = VO.run(firstFrameCnt=0, lastFrameCnt=1100)

plt.figure()
plt.plot(groundTruthPoses[:, 0, 3], groundTruthPoses[:, 2, 3], 'r',
         estimatedPoses_1x1_3000_BF[:, 0, 3], estimatedPoses_1x1_3000_BF[:, 2, 3], 'b',
         estimatedPoses_2x1_1500_BF[:, 0, 3], estimatedPoses_2x1_1500_BF[:, 2, 3], 'g',
         estimatedPoses_2x2_750_BF[:, 0, 3], estimatedPoses_2x2_750_BF[:, 2, 3], 'm',
         estimatedPoses_3x2_500_BF[:, 0, 3], estimatedPoses_3x2_500_BF[:, 2, 3], 'y')
plt.title(r"Trajectories with Brute-Force matcher", wrap=True)
plt.legend([r"Ground truth", r"1x1 3000", r"2x1 1500", r"2x2 750", r"3x2 500"])


print("Average frame processing time:")
print("\tFLANN 1x1 3000:", avgFrameProcessTime_1x1_3000_Flann)
print("\tFLANN 2x1 1500:", avgFrameProcessTime_2x1_1500_Flann)
print("\tFLANN 2x2 750:", avgFrameProcessTime_2x2_750_Flann)
print("\tFLANN 3x2 500:", avgFrameProcessTime_3x2_500_Flann)
print("\tBrute-Force 1x1 3000:", avgFrameProcessTime_1x1_3000_BF)
print("\tBrute-Force 2x1 1500:", avgFrameProcessTime_2x1_1500_BF)
print("\tBrute-Force 2x2 750:", avgFrameProcessTime_2x2_750_BF)
print("\tBrute-Force 3x2 500:", avgFrameProcessTime_3x2_500_BF)


# Testing all 11 KITTI sequences using the Brute-Force matcher with 3x2 image grid
for i in range(11):
    VO = VisualOdometry(sequences[i])
    estimatedPoses, groundTruthPoses, _ = VO.run()
    plt.figure()
    plt.plot(groundTruthPoses[:, 0, 3], groundTruthPoses[:, 2, 3], 'r',
             estimatedPoses[:, 0, 3], estimatedPoses[:, 2, 3], 'b')
    plt.title(r"Trajectory", wrap=True)
    plt.legend([r"Ground truth", r"Estimated"])

plt.show()
