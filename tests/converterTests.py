import numpy as np


def converterTest1():
    n = 2

    A = np.array([[1, 2], [3, 4]])
    # print(A)
    # newA = np.repeat(A, n ** 2)
    # print(newA)
    # newA = np.reshape(newA, (len(A) * n, len(A) * n))
    # print(newA)

    # Af = np.reshape(A, (-1,)).tolist()
    Anew = np.kron([A], np.ones((2, 2)))[0]
    print(Anew)


converterTest1()
