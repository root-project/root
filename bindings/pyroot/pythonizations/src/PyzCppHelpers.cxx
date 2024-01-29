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

#include "../../cppyy/CPyCppyy/src/CPPInstance.h"

// Call method with signature: obj->meth()
PyObject *CallPyObjMethod(PyObject *obj, const char *meth)
{
   return PyObject_CallMethod(obj, meth, "");
}

// Call method with signature: obj->meth(arg1)
PyObject *CallPyObjMethod(PyObject *obj, const char *meth, PyObject *arg1)
{
   return PyObject_CallMethod(obj, meth, "O", arg1);
}

// Convert generic python object into a boolean value
PyObject *BoolNot(PyObject *value)
{
   if (PyObject_IsTrue(value) == 1) {
      Py_DECREF(value);
      Py_RETURN_FALSE;
   } else {
      Py_XDECREF(value);
      Py_RETURN_TRUE;
   }
}

// Get the TClass of the C++ object proxied by pyobj
TClass *GetTClass(const PyObject *pyobj)
{
   return TClass::GetClass(GetScopedFinalNameFromPyObject(pyobj).c_str());
}

std::string GetScopedFinalNameFromPyObject(const PyObject *pyobj)
{
   return Cppyy::GetScopedFinalName(((CPyCppyy::CPPInstance*)pyobj)->ObjectIsA());
}
