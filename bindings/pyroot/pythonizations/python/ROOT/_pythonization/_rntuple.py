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
    # key can be either a RFieldToken already or a string. In the latter case, get a token to use it twice.
    if (
        not hasattr(type(key), "__cpp_name__")
        or type(key).__cpp_name__ != "ROOT::Experimental::REntry::RFieldToken"
    ):
        key = self.GetToken(key)
    fieldType = self.GetTypeName(key)
    return self._GetPtr[fieldType](key)


def _REntry_getitem(self, key):
    ptr = self.GetPtr(key)
    return ptr.get()[0]


def _REntry_setitem(self, key, value):
    ptr = self.GetPtr(key)
    ptr.get()[0] = value


@pythonization("REntry", ns="ROOT::Experimental")
def pythonize_REntry(klass):
    klass._GetPtr = klass.GetPtr
    klass.GetPtr = _REntry_GetPtr

    klass.__getitem__ = _REntry_getitem
    klass.__setitem__ = _REntry_setitem


def _RNTupleModel_CreateBare(*args):
    if len(args) >= 1:
        raise ValueError("no support for passing explicit RFieldZero")
    import ROOT

    return ROOT.Experimental.RNTupleModel._CreateBare()


def _RNTupleModel_GetDefaultEntry(self):
    raise RuntimeError("default entries are not supported in Python, call CreateEntry")


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

    klass.GetDefaultEntry = _RNTupleModel_GetDefaultEntry

    klass.MakeField = MethodTemplateGetter(klass.MakeField, _RNTupleModel_MakeField)


def _RNTupleReader_LoadEntry(self, *args):
    if len(args) < 2:
        raise ValueError(
            "default entries are not supported in Python, pass explicit entry"
        )
    return self._LoadEntry(*args)


@pythonization("RNTupleReader", ns="ROOT::Experimental")
def pythonize_RNTupleReader(klass):
    klass._LoadEntry = klass.LoadEntry
    klass.LoadEntry = _RNTupleReader_LoadEntry


def _RNTupleWriter_Fill(self, *args):
    if len(args) < 1:
        raise ValueError(
            "default entries are not supported in Python, pass explicit entry"
        )
    return self._Fill(*args)


@pythonization("RNTupleWriter", ns="ROOT::Experimental")
def pythonize_RNTupleWriter(klass):
    klass._Fill = klass.Fill
    klass.Fill = _RNTupleWriter_Fill
