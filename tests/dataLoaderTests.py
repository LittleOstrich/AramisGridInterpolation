import os

import paths
from data.dataLoader import getAllPathes
import unittest


class TestDataLoader(unittest.TestCase):

    def test_getAllPathes(self):
        sd = paths.projectBase
        ff = os.path.isdir
        allPathes = getAllPathes(sd=sd, filterFuntion=ff, debug=False)
        ds = allPathes[0]
        fs = allPathes[1]
        assert len(fs) == 0
        assert len(ds) != 0

        for d in ds:
            print(d)


if __name__ == '__main__':
    unittest.main()
