
#ifndef PYROOT_PYSTRINGS_H
#define PYROOT_PYSTRINGS_H

#include "Python.h"
#include "DllImport.h"

namespace PyROOT {

// python strings kept for performance reasons

namespace PyStrings {

R__EXTERN PyObject *gBranch;
R__EXTERN PyObject *gFitFCN;
R__EXTERN PyObject *gROOTns;
R__EXTERN PyObject *gSetBranchAddress;
R__EXTERN PyObject *gSetFCN;
R__EXTERN PyObject *gTClassDynCast;

} // namespace PyStrings

bool CreatePyStrings();
PyObject *DestroyPyStrings();

} // namespace PyROOT

#endif // !PYROOT_PYSTRINGS_H
