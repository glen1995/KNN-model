import cv2
import numpy as np 
from matplotlib import pyplot as plt 
# np.random.seed(2)
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
responses = np.random.randint(0,2,(25,1)).astype(np.float32)

red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')

blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],20,'b','o')

newcomer = np.random.randint(0,100,(10,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')
# print(trainData)

knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,responses)
ret,results,neighbours,dist = knn.findNearest(newcomer, 7)

print("result: ", results,"\n")
print("neighbours: ", neighbours,"\n")
print("distance: ", dist)

# plt.show()