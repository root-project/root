// Author: Danilo Piparo, Stefan Wunsch, Massimiliano Galli CERN  08/2018
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "CPyCppyy.h"
#include "CPPInstance.h"
#include "PyROOTPythonize.h"
#include "ProxyWrappers.h"
#include "Utility.h"
#include "PyzCppHelpers.hxx"
#include "TClass.h"
#include "TDirectory.h"
#include "TKey.h"
#include "TObject.h"
#include "Python.h"

using namespace CPyCppyy;

////////////////////////////////////////////////////////////////////////////////
/// \brief Implements the WriteObject method of TDirectory
/// This method allows to write objects into TDirectory instances with this
/// syntax:
/// ~~~{.py}
/// myDir.WriteObject(myObj, "myKeyName")
/// ~~~
PyObject *TDirectoryWriteObject(CPPInstance *self, PyObject *args)
{
   CPPInstance *wrt = nullptr;
   PyObject *name = nullptr;
   PyObject *option = nullptr;
   Int_t bufsize = 0;
   if (!PyArg_ParseTuple(args, const_cast<char *>("O!O!|O!i:TDirectory::WriteObject"),
                         &CPPInstance_Type, &wrt,
                         &CPyCppyy_PyText_Type, &name,
                         &CPyCppyy_PyText_Type, &option,
                         &bufsize))
      return nullptr;
   auto dir = (TDirectory *)GetTClass(self)->DynamicCast(TDirectory::Class(), self->GetObject());
   if (!dir) {
      PyErr_SetString(PyExc_TypeError,
                      "TDirectory::WriteObject must be called with a TDirectory instance as first argument");
      return nullptr;
   }

   // Implement a check on whether the object is derived from TObject or not. Similarly to what is done in
   // TDirectory::WriteObject with SFINAE. Unfortunately, 'wrt' is a void* in this scope and can't be casted to its
   // concrete type.
   auto *wrtclass = GetTClass(wrt);
   void *wrtobj = wrt->GetObject();
   Int_t result = 0;

   if (wrtclass->IsTObject()) {
      // If the found TClass is derived from TObject, cast the object to a TObject since we are just interested in the
      // object title for the purposes of the WriteTObject function.
      auto objtowrite = static_cast<TObject *>(wrtclass->DynamicCast(TObject::Class(), wrtobj));

      if (option != nullptr) {
         result =
            dir->WriteTObject(objtowrite, CPyCppyy_PyText_AsString(name), CPyCppyy_PyText_AsString(option), bufsize);
      } else {
         result = dir->WriteTObject(objtowrite, CPyCppyy_PyText_AsString(name));
      }
   } else {
      if (option != nullptr) {
         result = dir->WriteObjectAny(wrtobj, wrtclass, CPyCppyy_PyText_AsString(name),
                                      CPyCppyy_PyText_AsString(option), bufsize);
      } else {
         result = dir->WriteObjectAny(wrtobj, wrtclass, CPyCppyy_PyText_AsString(name));
      }
   }

   return PyInt_FromLong((Long_t)result);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add pythonisation of TDirectory::WriteObject
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
PyObject *PyROOT::AddDirectoryWritePyz(PyObject * /* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);
   Utility::AddToClass(pyclass, "WriteObject", (PyCFunction)TDirectoryWriteObject);
   Py_RETURN_NONE;
}
