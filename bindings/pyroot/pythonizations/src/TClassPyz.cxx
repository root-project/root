// Author: Enric Tejedor CERN  02/2019
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
#include "../../cppyy/CPyCppyy/src/Utility.h"

#include "PyROOTPythonize.h"

// ROOT
#include "TClass.h"

using namespace CPyCppyy;

namespace PyROOT{
void GetBuffer(PyObject *pyobject, void *&buf);
}

// Cast the void* returned by TClass::DynamicCast to the right type
PyObject *TClassDynamicCastPyz(PyObject *self, PyObject *args)
{
   // Parse arguments
   PyObject *pyclass = nullptr;
   PyObject *pyobject = nullptr;
   int up = 1;
   if (!PyArg_ParseTuple(args, "OO|i:DynamicCast", &pyclass, &pyobject, &up))
      return nullptr;

   if (!CPyCppyy::Instance_Check(pyclass)) {
      PyErr_Format(PyExc_TypeError,
         "DynamicCast argument 1 must be a cppyy instance, got '%.200s'",
         Py_TYPE(pyclass)->tp_name);
      return nullptr;
   }

   // Perform actual cast - calls default implementation of DynamicCast
   TClass *cl1 = (TClass *)CPyCppyy::Instance_AsVoidPtr(self);
   TClass *cl2 = (TClass *)CPyCppyy::Instance_AsVoidPtr(pyclass);

   void *address = cl1->DynamicCast(cl2, CPyCppyy::Instance_AsVoidPtr(pyobject), up);

   if (CPyCppyy::Instance_Check(pyobject)) {
      address = CPyCppyy::Instance_AsVoidPtr(pyobject);
   } else if (PyInt_Check(pyobject) || PyLong_Check(pyobject)) {
      address = (void *)PyLong_AsLongLong(pyobject);
   } else {
      PyROOT::GetBuffer(pyobject, address);
   }

   // Now use binding to return a usable class. Upcast: result is a base.
   // Downcast: result is a derived.
   TClass *tcl = TClass::GetClass(CPyCppyy::Instance_GetScopedFinalName(up ? pyclass : self).c_str());
   TClass *klass = (TClass *)tcl->DynamicCast(TClass::Class(), up ? CPyCppyy::Instance_AsVoidPtr(pyclass) : cl1);

   return CPyCppyy::Instance_FromVoidPtr(address, klass->GetName());
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add pythonization for TClass::DynamicCast.
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
/// received from Python.
///
/// TClass::DynamicCast returns a void* that the user still has to cast (it
/// will have the proper offset, though). Fix this by providing the requested
/// binding if the cast succeeded.
PyObject *PyROOT::AddTClassDynamicCastPyz(PyObject * /* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);
   Utility::AddToClass(pyclass, "DynamicCast", (PyCFunction)TClassDynamicCastPyz);
   Py_RETURN_NONE;
}
