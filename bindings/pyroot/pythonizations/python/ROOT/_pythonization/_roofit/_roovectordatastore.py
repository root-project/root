# Authors:
# * Jonas Rembser 07/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


class RooVectorDataStore(object):
    def to_numpy(self, copy=True):
        import numpy as np

        data = {}

        def check_for_duplicate_name(name):
            if name in data:
                raise RuntimeError("Attempt to add " + name + " twice to the numpy arrays!")

        array_info = self.getArrays()
        n = array_info.size

        for x in array_info.reals:
            check_for_duplicate_name(x.name)
            data[x.name] = np.frombuffer(x.data, dtype=np.float64, count=n)
        for x in array_info.cats:
            check_for_duplicate_name(x.name)
            data[x.name] = np.frombuffer(x.data, dtype=np.int32, count=n)
        if copy:
            for name in data.keys():
                data[name] = np.copy(data[name])

        return data
