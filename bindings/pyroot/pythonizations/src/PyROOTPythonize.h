// Author: Enric Tejedor CERN  06/2018
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef PYROOT_PYTHONIZE_H
#define PYROOT_PYTHONIZE_H

#include "Python.h"

namespace PyROOT {

PyObject *AddCPPInstancePickling(PyObject *self, PyObject *args);

PyObject *AddPrettyPrintingPyz(PyObject *self, PyObject *args);

PyObject *GetBranchAttr(PyObject *self, PyObject *args);
PyObject *BranchPyz(PyObject *self, PyObject *args);

PyObject *AddTClassDynamicCastPyz(PyObject *self, PyObject *args);

PyObject *AddTObjectEqNePyz(PyObject *self, PyObject *args);

PyObject *CPPInstanceExpand(PyObject *self, PyObject *args);

} // namespace PyROOT

#endif // !PYROOT_PYTHONIZE_H
