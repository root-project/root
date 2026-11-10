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


def _is_operation(op):
    import ROOT

    return isinstance(op, ROOT.uhi.rebin) or op == ROOT.uhi.sum or op == sum


def _translate_operation(op):
    """Translate a single Python "object" representing an operation into an object for RSliceSpec."""
    import ROOT

    if isinstance(op, ROOT.uhi.rebin):
        return ROOT.Experimental.RSliceSpec.ROperationRebin(op.ngroup)
    elif op == ROOT.uhi.sum or op == sum:
        return ROOT.Experimental.RSliceSpec.ROperationSum()
    else:
        raise TypeError(f"unsupported operation type: {type(op).__name__}")


def _translate_slice(self, i, axis, s):
    """Translate a single Python slice into an RSliceSpec."""
    import ROOT

    RSliceSpec = ROOT.Experimental.RSliceSpec

    if _is_index(s):
        index = _translate_index(s)
        if not axis.IsValidIndex(index):
            raise IndexError(f"invalid bin index for axis {i}: {s}")
        if not index.IsNormal():
            raise IndexError("normal bin required in slicing")

        # A single value v is like v:v+1:sum
        r = axis.GetNormalRange(index, index + 1)
        return RSliceSpec(r, RSliceSpec.ROperationSum())

    elif _is_operation(s):
        operation = _translate_operation(s)
        return RSliceSpec(operation)

    elif isinstance(s, slice):
        begin = None
        if s.start is not None:
            begin = _translate_index(s.start)
            if not begin.IsNormal():
                raise IndexError("normal bin required in slicing")

        end = None
        if s.stop is not None:
            if s.stop == len:
                end = ROOT.Experimental.RBinIndex(axis.GetNNormalBins())
            else:
                end = _translate_index(s.stop)
                if not end.IsNormal():
                    raise IndexError("normal bin required in slicing")

        operation = None
        if s.step is not None:
            operation = _translate_operation(s.step)

        if begin is None and end is None:
            if operation is None:
                return RSliceSpec()
            else:
                return RSliceSpec(operation)

        # At least one endpoint is given.
        elif operation is not None and type(operation).__cpp_name__ == "ROOT::Experimental::RSliceSpec::ROperationSum":
            # For sums it makes a difference whether flow bins are included. If only one endpoint is given, we cannot guess.
            if begin is None or end is None:
                raise IndexError("either none or both endpoints required for sum")

            r = axis.GetNormalRange(begin, end)
            return RSliceSpec(r, operation)

        # For other slice operations, it does not make a difference whether the range includes the flow bins.
        elif begin is None:
            begin = ROOT.Experimental.RBinIndex(0)
        elif end is None:
            end = ROOT.Experimental.RBinIndex(axis.GetNNormalBins())

        r = axis.GetNormalRange(begin, end)
        if operation is None:
            return RSliceSpec(r)
        else:
            return RSliceSpec(r, operation)

    else:
        raise TypeError(f"unsupported index type: {type(s).__name__}")


def _translate_slices(self, slices):
    """Translate a tuple of Python slices into a std::vector<RSliceSpec>."""
    # TODO: expand ellipsis and dictionaries

    if len(slices) != self.GetNDimensions():
        raise IndexError(f"invalid number of slices: expected {self.GetNDimensions()}, but got {len(slices)}")

    import ROOT

    axes = self.GetAxes()
    vec = ROOT.std.vector["ROOT::Experimental::RSliceSpec"]()
    for i, s in enumerate(slices):
        sliceSpec = _translate_slice(self, i, axes[i], s)
        vec.push_back(sliceSpec)
    return vec


def _ensure_tuple(index):
    """Ensure that the passed Python index is wrapped in a tuple."""
    return (index,) if not isinstance(index, tuple) else index


def _getitem(self, index):
    indices = _ensure_tuple(index)
    if _is_tuple_of_indices(indices):
        indices = _translate_indices(self, indices)
        return self.GetBinContent(indices)
    else:
        slices = _translate_slices(self, indices)
        return self.Slice(slices)


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
