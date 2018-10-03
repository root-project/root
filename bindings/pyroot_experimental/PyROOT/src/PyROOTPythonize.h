
#ifndef PYROOT_PYTHONIZE_H
#define PYROOT_PYTHONIZE_H

struct _object;
typedef _object PyObject;

namespace PyROOT {

PyObject *AddPrettyPrintingPyz(PyObject *self, PyObject *args);
PyObject *AddBranchAttrSyntax(PyObject *self, PyObject *args);
PyObject *SetBranchAddressPyz(PyObject *self, PyObject *args);
PyObject *AddDirectoryWritePyz(PyObject *self, PyObject *args);
PyObject *AddDirectoryAttrSyntaxPyz(PyObject *self, PyObject *args);
PyObject *AddFileOpenPyz(PyObject *self, PyObject *args);
PyObject *PythonizeTTree(PyObject *self, PyObject *args);
PyObject *GetEndianess(PyObject *self);
PyObject *GetVectorDataPointer(PyObject *self, PyObject *args);
PyObject *GetSizeOfType(PyObject *self, PyObject *args);

} // namespace PyROOT

#endif // !PYROOT_PYTHONIZE_H
