from libROOTPython import GetEndianess, GetVectorDataPointer, GetSizeOfType, AddRTensorGetSetItem
from ROOT import pythonization
from ROOT.pythonization._rvec import _array_interface_dtypes, _array_interface_dtype_map


def get_array_interface(self):
    cppname = type(self).__cppname__
    for dtype in _array_interface_dtypes:
        if cppname.endswith("<{}>".format(dtype)):
            dtype_numpy = _array_interface_dtype_map[dtype]
            dtype_size = GetSizeOfType(dtype)
            endianess = GetEndianess()
            shape = self.GetShape()
            rvec = self.GetDataAsVec()
            pointer = GetVectorDataPointer(rvec, type(rvec).__cppname__)
            return {
                "shape": tuple(shape),
                "typestr": "{}{}{}".format(endianess, dtype_numpy, dtype_size),
                "version": 3,
                "data": (pointer, False)
            }


def add_array_interface_property(klass, name):
    if True in [
            name.endswith("<{}>".format(dtype))
            for dtype in _array_interface_dtypes
    ]:
        klass.__array_interface__ = property(get_array_interface)


def add_getsetitem(klass, name):
    for dtype in _array_interface_dtypes:
        if name.endswith("<{}>".format(dtype)):
            AddRTensorGetSetItem(klass, dtype)


@pythonization
def pythonize_rtensor(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name.startswith("TMVA::Experimental::RTensor<"):
        # Add numpy array interface
        add_array_interface_property(klass, name)
        add_getsetitem(klass, name)

    return True
