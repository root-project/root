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

PyObject *CallPyObjMethod(PyObject *obj, const char *meth)
{
   // Helper; call method with signature: obj->meth()
   Py_INCREF(obj);
   PyObject* result = PyObject_CallMethod(obj, const_cast<char*>(meth), const_cast<char*>(""));
   Py_DECREF(obj);
   return result;
}

TClass *GetTClass(const CPyCppyy::CPPInstance *pyobj)
{
   return TClass::GetClass(Cppyy::GetFinalName(pyobj->ObjectIsA()).c_str());
}
