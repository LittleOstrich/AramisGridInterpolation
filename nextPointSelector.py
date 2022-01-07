from abc import ABC, abstractmethod

import numpy as np


class Selector(ABC):
    nextEvenPoints = None
    nextOddPoints = None
    evenVisited = None
    oddVisited = None
    mode = None
    N = None
    sn = None
    switch = -1
    iterationCounter = 0
    pointMap = None

    @abstractmethod
    def __init__(self, N):
        self.N = N
        self.sn = np.random.randint(N)
        self.pointMap = np.zeros((N, 2))
        self.pointMap[self.sn, 0] = 1000
        self.pointMap[self.sn, 1] = 1000

    @abstractmethod
    def nextPoint(self):
        assert False


class PriotizingSelector(Selector):
    def __init__(self, N):
        self.nextEvenPoints = set()
        self.nextOddPoints = set()
        self.evenVisited = set()
        self.oddVisited = set()
        self.evenVisited.add(self.sn)
        super().__init__(N)  # shouldn't that be called first (?)

    def nextPoint(self):
        pass


class BalancingSelector(Selector):

    def __init__(self, N):
        self.nextEvenPoints = list()
        self.nextOddPoints = list()
        self.evenVisited = set()
        self.oddVisited = set()
        self.evenVisited.add(self.sn)
        super().__init__(N)  # shouldn't that be called first (?)

    def nextPoint(self):
        n = len(self.evenVisited)
        m = len(self.oddVisited)

        qq = PriotizingSelector(1000)
        print(qq.N)
