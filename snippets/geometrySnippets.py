import numpy as np

from helpers.geometryHelper import computeTriangleArea, computeTriangleSidelengths, computeTriangleAngels

v1 = np.array([0, 0, 0])
v2 = np.array([0, 0, 1])
v3 = np.array([0, 1, 0])
l = [v1, v2, v3]
area = computeTriangleArea(*l)
s1, s2, s3 = computeTriangleSidelengths(*l)
alpha, beta, gamma = computeTriangleAngels(*l, fromPositionVectors=False)

print(area)
print(s1, s2, s3)
print(alpha, beta, gamma)
