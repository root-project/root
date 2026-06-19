from . import pythonization


def _is_RBinIndex(index):
    return hasattr(type(index), "__cpp_name__") and type(index).__cpp_name__ == "ROOT::Experimental::RBinIndex"


def _is_index(index):
    import ROOT

    # TODO: implement support for loc
    return isinstance(index, int) or _is_RBinIndex(index) or index == ROOT.uhi.underflow or index == ROOT.uhi.overflow


def _is_tuple_of_indices(indices):
    return all([_is_index(i) for i in indices])


def _translate_index(index):
    """Translate a single Python index into an RBinIndex."""
    import ROOT

    # TODO: implement support for loc

    if isinstance(index, int):
        return ROOT.Experimental.RBinIndex(index)
    elif _is_RBinIndex(index):
        return index
    elif index == ROOT.uhi.underflow:
        return ROOT.Experimental.RBinIndex.Underflow()
    elif index == ROOT.uhi.overflow:
        return ROOT.Experimental.RBinIndex.Overflow()
    else:
        raise TypeError(f"unsupported index type: {type(index).__name__}")


def _translate_indices(self, indices):
    """Translate a tuple of Python indices into a std::vector<RBinIndex>."""
    if len(indices) != self.GetNDimensions():
        raise IndexError(f"invalid number of indices: expected {self.GetNDimensions()}, but got {len(indices)}")

    import ROOT

    axes = self.GetAxes()
    vec = ROOT.std.vector["ROOT::Experimental::RBinIndex"]()
    for i, index in enumerate(indices):
        binIndex = _translate_index(index)
        if not axes[i].IsValidIndex(binIndex):
            raise IndexError(f"invalid bin index for axis {i}: {index}")
        vec.push_back(binIndex)
    return vec


def _ensure_tuple(index):
    """Ensure that the passed Python index is wrapped in a tuple."""
    return (index,) if not isinstance(index, tuple) else index


def _getitem(self, index):
    indices = _ensure_tuple(index)
    if not _is_tuple_of_indices(indices):
        raise ValueError("could not convert indices")
    indices = _translate_indices(self, indices)
    return self.GetBinContent(indices)


def _setitem(self, index, value):
    indices = _ensure_tuple(index)
    if not _is_tuple_of_indices(indices):
        raise ValueError("could not convert indices")
    indices = _translate_indices(self, indices)
    self.SetBinContent(indices, value)


@pythonization("RHistEngine<", ns="ROOT::Experimental", is_prefix=True)
def pythonize_engine(klass):
    klass.__getitem__ = _getitem
    klass.__setitem__ = _setitem


@pythonization("RHist<", ns="ROOT::Experimental", is_prefix=True)
def pythonize_hist(klass):
    klass.__getitem__ = _getitem
    klass.__setitem__ = _setitem
