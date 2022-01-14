import numpy as np


def asDisplacementVectors(a, b, c):
    v1, v2, v3 = a - b, b - c, c - a
    return v1, v2, v3


def computeTriangleArea(a, b, c, fromPositionVectors=True):
    if fromPositionVectors:
        a, b, c = asDisplacementVectors(a, b, c)
    s1 = a - b
    s2 = a - c
    area = 0.5 * np.linalg.norm(np.cross(s1, s2), ord=2)
    return area


def computeTriangleSidelengths(a, b, c, fromPositionVectors=True):
    if fromPositionVectors:
        s1 = np.linalg.norm(a - b, ord=2)  # euclidean norm
        s2 = np.linalg.norm(b - c, ord=2)
        s3 = np.linalg.norm(c - b, ord=2)
    else:
        s1 = np.linalg.norm(a, ord=2)  # euclidean norm
        s2 = np.linalg.norm(b, ord=2)
        s3 = np.linalg.norm(c, ord=2)
    return s1, s2, s3


def computeTriangleAngels(a, b, c, fromPositionVectors=True):
    if fromPositionVectors:
        v1 = a - b
        v2 = b - c
        v3 = a - c
        a = v1
        b = v2
        c = v3

    absA = np.linalg.norm(a, ord=2)
    absB = np.linalg.norm(b, ord=2)
    absC = np.linalg.norm(c, ord=2)

    alpha = np.arccos(np.dot(a, b) / (absA * absB))
    beta = np.arccos(np.dot(b, c) / (absB * absC))
    gamma = np.arccos(np.dot(c, a) / (absA * absC))

    x = np.array([alpha, beta, gamma]) * 180 / np.pi
    x = np.round(x, decimals=1)

    x[0] = np.min([x[0], 180 - x[0]])
    x[1] = np.min([x[1], 180 - x[1]])
    x[2] = 180 - x[0] - x[1]
    x = np.sort(x)
    return x[0], x[1], x[2]
