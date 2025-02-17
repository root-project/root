# Author: Jonas Hahnfeld CERN 11/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization
from ._pyz_utils import MethodTemplateGetter, MethodTemplateWrapper


def _REntry_GetPtr(self, key):
    raise RuntimeError("GetPtr is not supported in Python, use indexing")


def _REntry_CallGetPtr(self, key):
    # key can be either a RFieldToken already or a string. In the latter case, get a token to use it twice.
    if (
        not hasattr(type(key), "__cpp_name__")
        or type(key).__cpp_name__ != "ROOT::Experimental::REntry::RFieldToken"
    ):
        key = self.GetToken(key)
    fieldType = self.GetTypeName(key)
    return self._GetPtr[fieldType](key)


def _REntry_getitem(self, key):
    ptr_proxy = self._CallGetPtr(key)
    if type(ptr_proxy).__cpp_name__.startswith("std::shared_ptr"):
        return ptr_proxy.__deref__()
    # Otherwise, for non-fundamental types, cppyy already returns the pointee.
    return ptr_proxy


def _REntry_setitem(self, key, value):
    ptr_proxy = self._CallGetPtr(key)
    if type(ptr_proxy).__cpp_name__.startswith("std::shared_ptr"):
        ptr_proxy.get()[0] = value
    else:
        # Otherwise, for non-fundamental types, cppyy already returns the pointee.
        ptr_proxy.__assign__(value)


@pythonization("REntry", ns="ROOT::Experimental")
def pythonize_REntry(klass):
    klass._GetPtr = klass.GetPtr
    klass.GetPtr = _REntry_GetPtr
    klass._CallGetPtr = _REntry_CallGetPtr

    klass.__getitem__ = _REntry_getitem
    klass.__setitem__ = _REntry_setitem


def _RNTupleModel_CreateBare(*args):
    if len(args) >= 1:
        raise ValueError("no support for passing explicit RFieldZero")
    import ROOT

    return ROOT.Experimental.RNTupleModel._CreateBare()


def _RNTupleModel_CreateEntry(self):
    raise RuntimeError(
        "creating entries from model is not supported in Python, call CreateEntry on the reader or writer"
    )


def _RNTupleModel_GetDefaultEntry(self):
    raise RuntimeError(
        "default entries are not supported in Python, call CreateEntry on the reader or writer"
    )


class _RNTupleModel_MakeField(MethodTemplateWrapper):
    def __call__(self, *args):
        self._original_method(*args)
        # We do not support default entries in Python, so do not even return the nullptr.
        return


@pythonization("RNTupleModel", ns="ROOT::Experimental")
def pythonize_RNTupleModel(klass):
    # We do not support default entries in Python, so always create a bare model.
    klass.Create = _RNTupleModel_CreateBare
    klass._CreateBare = klass.CreateBare
    klass.CreateBare = _RNTupleModel_CreateBare

    klass.CreateBareEntry = _RNTupleModel_CreateEntry
    klass.CreateEntry = _RNTupleModel_CreateEntry
    klass.GetDefaultEntry = _RNTupleModel_GetDefaultEntry

    klass.MakeField = MethodTemplateGetter(klass.MakeField, _RNTupleModel_MakeField)


def _RNTupleReader_Open(maybe_model, *args):
    if (
        hasattr(type(maybe_model), "__cpp_name__")
        and type(maybe_model).__cpp_name__ == "ROOT::Experimental::RNTupleModel"
    ):
        # In Python, the user cannot create REntries directly from a model, so we can safely clone it and avoid destructively passing the user argument.
        maybe_model = maybe_model.Clone()
    import ROOT

    return ROOT.Experimental.RNTupleReader._Open(maybe_model, *args)


def _RNTupleReader_LoadEntry(self, *args):
    if len(args) < 2:
        raise ValueError(
            "default entries are not supported in Python, pass explicit entry"
        )
    return self._LoadEntry(*args)


@pythonization("RNTupleReader", ns="ROOT::Experimental")
def pythonize_RNTupleReader(klass):
    klass._Open = klass.Open
    klass.Open = _RNTupleReader_Open

    klass._LoadEntry = klass.LoadEntry
    klass.LoadEntry = _RNTupleReader_LoadEntry


def _RNTupleWriter_Append(model, *args):
    # In Python, the user cannot create REntries directly from a model, so we can safely clone it and avoid destructively passing the user argument.
    model = model.Clone()
    import ROOT

    return ROOT.Experimental.RNTupleWriter._Append(model, *args)


def _RNTupleWriter_Recreate(model_or_fields, *args):
    if (
        hasattr(type(model_or_fields), "__cpp_name__")
        and type(model_or_fields).__cpp_name__ == "ROOT::Experimental::RNTupleModel"
    ):
        # In Python, the user cannot create REntries directly from a model, so we can safely clone it and avoid destructively passing the user argument.
        model_or_fields = model_or_fields.Clone()
    import ROOT

    return ROOT.Experimental.RNTupleWriter._Recreate(model_or_fields, *args)


def _RNTupleWriter_Fill(self, *args):
    if len(args) < 1:
        raise ValueError(
            "default entries are not supported in Python, pass explicit entry"
        )
    return self._Fill(*args)


def _RNTupleWriter_exit(self, *args):
    self.CommitDataset()
    return False


@pythonization("RNTupleWriter", ns="ROOT::Experimental")
def pythonize_RNTupleWriter(klass):
    klass._Append = klass.Append
    klass.Append = _RNTupleWriter_Append
    klass._Recreate = klass.Recreate
    klass.Recreate = _RNTupleWriter_Recreate

    klass._Fill = klass.Fill
    klass.Fill = _RNTupleWriter_Fill

    klass.__enter__ = lambda writer: writer
    klass.__exit__ = _RNTupleWriter_exit
