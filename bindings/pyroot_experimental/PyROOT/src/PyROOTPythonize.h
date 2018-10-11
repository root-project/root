
#ifndef PYROOT_PYTHONIZE_H
#define PYROOT_PYTHONIZE_H

#include "Python.h"

namespace PyROOT {

PyObject *AddPrettyPrintingPyz(PyObject *self, PyObject *args);
PyObject *AddBranchAttrSyntax(PyObject *self, PyObject *args);
PyObject *SetBranchAddressPyz(PyObject *self, PyObject *args);
PyObject *GetEndianess(PyObject *self);
PyObject *GetVectorDataPointer(PyObject *self, PyObject *args);
PyObject *GetSizeOfType(PyObject *self, PyObject *args);

} // namespace PyROOT

#endif // !PYROOT_PYTHONIZE_H
