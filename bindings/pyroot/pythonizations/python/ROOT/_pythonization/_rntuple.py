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


def _try_getptr(entry, fieldType, key):
    rentry_getptr_typeerrors = []
    try:
        return entry._GetPtr[fieldType](key)
    except TypeError as e:
        # The input field type name might not be found when trying to instantiate
        # the template via the Python bindings. At the moment, there is one
        # notable case when this happens. If the type name corresponds to a class
        # that has alternative names registered via its dictionary, the name
        # passed by the user may not be the fully qualified name seen by the
        # compiler. We tap into the knowledge of TClassTable to look for all the
        # alternative names, and try if there is a correspondence with one of those.
        # Note: clearly a prerequisite for the following section to work is that
        # the dictionary of the class was loaded and its alternative names were
        # registered. This currently happens at loading time of the RNTupleModel
        # which loads the classes found in the schema and also during the first
        # attempt at calling _GetPtr with the user-provided class name (in
        # particular when it tries to instantiate the function template)
        rentry_getptr_typeerrors.append(e)
        import ROOT

        alt_field_type_names = ROOT.TClassTable.GetClassAlternativeNames(fieldType)
        for alt_field_type_name in alt_field_type_names:
            try:
                # Need to convert std::string to Python string
                return entry._GetPtr[str(alt_field_type_name)](key)
            except TypeError as alt_e:
                rentry_getptr_typeerrors.append(alt_e)

    err_msg = f"Failed to retrieve entry value for field type name '{fieldType}'. Full stack trace follows:\n"
    for ex in rentry_getptr_typeerrors:
        err_msg += str(ex) + "\n"

    raise TypeError(err_msg)


def _REntry_CallGetPtr(self, key):
    # key can be either a RFieldToken already or a string. In the latter case, get a token to use it twice.
    if not hasattr(type(key), "__cpp_name__") or type(key).__cpp_name__ != "ROOT::RFieldToken":
        key = self.GetToken(key)
    fieldType = self.GetTypeName(key)
    return _try_getptr(self, fieldType, key)


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


@pythonization("REntry", ns="ROOT")
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

    return ROOT.RNTupleModel._CreateBare()


def _RNTupleModel_CreateEntry(self):
    raise RuntimeError(
        "creating entries from model is not supported in Python, call CreateEntry on the reader or writer"
    )


def _RNTupleModel_GetDefaultEntry(self):
    raise RuntimeError("default entries are not supported in Python, call CreateEntry on the reader or writer")


class _RNTupleModel_MakeField(MethodTemplateWrapper):
    def __call__(self, *args):
        self._original_method(*args)
        # We do not support default entries in Python, so do not even return the nullptr.
        return


@pythonization("RNTupleModel", ns="ROOT")
def pythonize_RNTupleModel(klass):
    # We do not support default entries in Python, so always create a bare model.
    klass.Create = _RNTupleModel_CreateBare
    klass._CreateBare = klass.CreateBare
    klass.CreateBare = _RNTupleModel_CreateBare

    klass.CreateBareEntry = _RNTupleModel_CreateEntry
    klass.CreateEntry = _RNTupleModel_CreateEntry
    klass.GetDefaultEntry = _RNTupleModel_GetDefaultEntry

    klass.MakeField = MethodTemplateGetter(klass.MakeField, _RNTupleModel_MakeField)


# Wrapper class used for RNTupleReader and RNTupleWriter.
# It deletes the underlying smart pointer on context manager exit and ensures that the inner object becomes
# inaccessible by raising an error every time an attribute of the object is accessed.
# It also raises an error if `with` statements using the same object are nested.
# This is a generic class and can in principle be used with any class that needs this behavior.
class RNTupleContextWrapper:
    def __init__(self, inner, pretty_name, on_ctx_enter = None, on_ctx_exit = None):
        self._inner = inner
        self._pretty_name = pretty_name
        self._closed = False
        self._in_context = False
        self._on_ctx_enter = on_ctx_enter
        self._on_ctx_exit = on_ctx_exit

    def __getattribute__(self, name):
        if name.startswith('_'):
            return super().__getattribute__(name)

        if super().__getattribute__("_closed"):
            raise RuntimeError(
                f"cannot access {super().__getattribute__('_pretty_name')} after the `with` statement is exited"
            )
        return super().__getattribute__("_inner").__getattribute__(name)

    def __enter__(self, *args):
        if self._on_ctx_enter:
            self._on_ctx_enter(self._inner)
        if self._closed:
            raise RuntimeError(f"cannot reuse {self._pretty_name} in multiple `with` statements")
        if self._in_context:
            raise RuntimeError(f"cannot nest `with` statements using the same {self._pretty_name}")

        self._in_context = True
        return self

    def __exit__(self, *args):
        assert self._in_context and not self._closed
        if self._on_ctx_exit:
            self._on_ctx_exit(self._inner)
        self._in_context = False
        self._closed = True
        self._inner.__smartptr__().reset()
        return False
   

def _RNTupleReader_Open(maybe_model, *args):
    if hasattr(type(maybe_model), "__cpp_name__") and type(maybe_model).__cpp_name__ == "ROOT::RNTupleModel":
        # In Python, the user cannot create REntries directly from a model, so we can safely clone it and avoid destructively passing the user argument.
        maybe_model = maybe_model.Clone()
    import ROOT

    return RNTupleContextWrapper(ROOT.RNTupleReader._Open(maybe_model, *args), "RNTupleReader")


def _RNTupleReader_LoadEntry(self, *args):
    if len(args) < 2:
        raise ValueError("default entries are not supported in Python, pass explicit entry")
    return self._LoadEntry(*args)


@pythonization("RNTupleReader", ns="ROOT")
def pythonize_RNTupleReader(klass):
    klass._Open = klass.Open
    klass.Open = _RNTupleReader_Open

    klass._LoadEntry = klass.LoadEntry
    klass.LoadEntry = _RNTupleReader_LoadEntry


def _RNTupleWriter_Append(model, *args):
    # In Python, the user cannot create REntries directly from a model, so we can safely clone it and avoid destructively passing the user argument.
    model = model.Clone()
    import ROOT

    return RNTupleContextWrapper(ROOT.RNTupleWriter._Append(model, *args), "RNTupleWriter", on_ctx_exit = _RNTupleWriter_exit)


def _RNTupleWriter_Recreate(model_or_fields, *args):
    if hasattr(type(model_or_fields), "__cpp_name__") and type(model_or_fields).__cpp_name__ == "ROOT::RNTupleModel":
        # In Python, the user cannot create REntries directly from a model, so we can safely clone it and avoid destructively passing the user argument.
        model_or_fields = model_or_fields.Clone()
    import ROOT

    return RNTupleContextWrapper(ROOT.RNTupleWriter._Recreate(model_or_fields, *args), "RNTupleWriter", on_ctx_exit = _RNTupleWriter_exit)


def _RNTupleWriter_Fill(self, *args):
    if len(args) < 1:
        raise ValueError("default entries are not supported in Python, pass explicit entry")
    return self._Fill(*args)


def _RNTupleWriter_exit(self):
    self.CommitDataset()


@pythonization("RNTupleWriter", ns="ROOT")
def pythonize_RNTupleWriter(klass):
    klass._Append = klass.Append
    klass.Append = _RNTupleWriter_Append
    klass._Recreate = klass.Recreate
    klass.Recreate = _RNTupleWriter_Recreate

    klass._Fill = klass.Fill
    klass.Fill = _RNTupleWriter_Fill
