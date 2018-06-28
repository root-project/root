
#ifndef PYROOT_PYTHONIZE_H
#define PYROOT_PYTHONIZE_H

#include "Python.h"

namespace PyROOT {

PyObject *PythonizeGeneric(PyObject *self, PyObject *args);
PyObject *PythonizeTTree(PyObject *self, PyObject *args);

} // namespace PyROOT

#endif // !PYROOT_PYTHONIZE_H
