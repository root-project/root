// Author: the ROOT team, CERN

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPython_PyGILRAII
#define ROOT_TPython_PyGILRAII

#include <Python.h>

/// Acquire the Python GIL for the current scope.
/// See https://docs.python.org/3/c-api/init.html#non-python-created-threads
class PyGILRAII {
   PyGILState_STATE fGILState;

public:
   PyGILRAII() : fGILState(PyGILState_Ensure()) {}
   ~PyGILRAII() { PyGILState_Release(fGILState); }
};

#endif // ROOT_TPython_PyGILRAII
