// Author: Enric Tejedor CERN  05/2019
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Bindings
#include "CPyCppyy/API.h"

#include "../../cppyy/CPyCppyy/src/CPyCppyy.h"
#include "../../cppyy/CPyCppyy/src/CPPInstance.h"
#include "../../cppyy/CPyCppyy/src/Utility.h"

#include "PyROOTPythonize.h"

// ROOT
#include "TObject.h"

namespace {

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

// Call method with signature: obj->meth(arg1)
PyObject *CallPyObjMethod(PyObject *obj, const char *meth, PyObject *arg1)
{
   return PyObject_CallMethod(obj, meth, "O", arg1);
}

} // namespace

using namespace CPyCppyy;

// Implement Python's __eq__ with TObject::IsEqual
PyObject *TObjectIsEqual(PyObject *self, PyObject *obj)
{
   if (!CPyCppyy::Instance_Check(obj) || !((CPPInstance *)obj)->fObject)
      return CPPInstance_Type.tp_richcompare(self, obj, Py_EQ);

   return CallPyObjMethod(self, "IsEqual", obj);
}

// Implement Python's __ne__ with TObject::IsEqual
PyObject *TObjectIsNotEqual(PyObject *self, PyObject *obj)
{
   if (!CPyCppyy::Instance_Check(obj) || !((CPPInstance *)obj)->fObject)
      return CPPInstance_Type.tp_richcompare(self, obj, Py_NE);

   return BoolNot(CallPyObjMethod(self, "IsEqual", obj));
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add pythonization for equality and inequality operators in
///        TObject
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
/// received from Python.
///
/// The equality and inequality operators are better implemented in C++,
/// since we need to need to rely on Cppyy's rich comparison if the object
/// we are comparing ourselves with is not a Python proxy or if it contains
/// a null pointer. For example, we need to support the comparison to None.
///
/// The rest of comparison operators (i.e. those that define order)
/// can be implemented in Python, throwing a NotImplemented exception
/// if we are not comparing two proxies to TObject or derivate.
PyObject *PyROOT::AddTObjectEqNePyz(PyObject * /* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);
   Utility::AddToClass(pyclass, "__eq__", (PyCFunction)TObjectIsEqual, METH_O);
   Utility::AddToClass(pyclass, "__ne__", (PyCFunction)TObjectIsNotEqual, METH_O);
   Py_RETURN_NONE;
}
