// Author: Danilo Piparo CERN  08/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**

Set of helper functions that are invoked from the C++ implementation of
pythonizations.

*/

#include "PyzCppHelpers.hxx"

#include "CPyCppyy.h"
#include "CPPInstance.h"
#include "TClass.h"

// Call method with signature: obj->meth()
PyObject *CallPyObjMethod(PyObject *obj, const char *meth)
{
   return PyObject_CallMethod(obj, const_cast<char *>(meth), const_cast<char *>(""));
}

// Call method with signature: obj->meth(arg1)
PyObject *CallPyObjMethod(PyObject *obj, const char *meth, PyObject *arg1)
{
   return PyObject_CallMethod(obj, const_cast<char *>(meth), const_cast<char *>("O"), arg1);
}

// Convert generic python object into a boolean value
PyObject *BoolNot(PyObject *value)
{
   if (PyObject_IsTrue(value) == 1) {
      Py_INCREF(Py_False);
      Py_DECREF(value);
      return Py_False;
   } else {
      Py_INCREF(Py_True);
      Py_XDECREF(value);
      return Py_True;
   }
}

// Get the TClass of the C++ object proxied by pyobj
TClass *GetTClass(const CPyCppyy::CPPInstance *pyobj)
{
   return TClass::GetClass(Cppyy::GetFinalName(pyobj->ObjectIsA()).c_str());
}
