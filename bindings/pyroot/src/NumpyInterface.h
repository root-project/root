// @(#)root/pyroot:$Id$
// Author: Jim Pivarski, Jul 2017

#ifndef PYROOT_NUMPYITERATOR_H
#define PYROOT_NUMPYITERATOR_H

#include <Python.h>

namespace PyROOT {

#if PY_VERSION_HEX >= 0x03000000
  void* InitializeNumpy();
#else
  void InitializeNumpy();
#endif

  PyObject* FillNumpyArray(PyObject* self, PyObject* args);

} // namespace PyROOT

#endif // PYROOT_NUMPYITERATOR_H
