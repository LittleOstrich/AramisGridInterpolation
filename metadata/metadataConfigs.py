class viewPointMapMetadata:
    hexOnly = "hexagonsOnly"

    @staticmethod
    def hexOnlyVals():
        vals = [False, True]
        return vals

    @staticmethod
    def configs():
        ret = list()

        for v in viewPointMapMetadata.hexOnlyVals():
            d = dict()
            d[viewPointMapMetadata.hexOnly] = v
            ret.append(d)
        return ret
