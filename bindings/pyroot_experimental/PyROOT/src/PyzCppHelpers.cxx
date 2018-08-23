// Author: Danilo Piparo CERN  08/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "PyzCppHelpers.hxx"

#include "CPyCppyy.h"
#include "CPPInstance.h"
#include "TClass.h"

PyObject* CallPyObjMethod(PyObject* obj, const char* meth, PyObject* arg1)
{
// Helper; call method with signature: obj->meth(arg1).
    Py_INCREF(obj);
    PyObject* result = PyObject_CallMethod(
        obj, const_cast<char*>(meth), const_cast<char*>("O"), arg1);
    Py_DECREF(obj);
    return result;
}

TClass* OP2TCLASS( CPyCppyy::CPPInstance* pyobj ) {
   return TClass::GetClass( Cppyy::GetFinalName( pyobj->ObjectIsA() ).c_str());
}
