// Author: Danilo Piparo, Stefan Wunsch CERN  08/2018
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
#include "CallContext.h"
#include "PyROOTPythonize.h"
#include "ProxyWrappers.h"
#include "TPython.h"
#include "Utility.h"
#include "PyzCppHelpers.hxx"
#include "TClass.h"
#include "TDirectory.h"
#include "TKey.h"
#include "Python.h"

using namespace CPyCppyy;

////////////////////////////////////////////////////////////////////////////////
/// \brief Implements the WriteObject method of TDirectory
/// This method allows to write objects into TDirectory instances with this
/// syntax:
/// ~~~{.python}
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
                         &CPyCppyy_PyUnicode_Type, &name,
                         &CPyCppyy_PyUnicode_Type, &option,
                         &bufsize))
      return nullptr;
   auto dir = (TDirectory *)OP2TCLASS(self)->DynamicCast(TDirectory::Class(), self->GetObject());
   if (!dir) {
      PyErr_SetString(PyExc_TypeError,
                      "TDirectory::WriteObject must be called with a TDirectory instance as first argument");
      return nullptr;
   }
   Int_t result = 0;
   if (option != nullptr) {
      result = dir->WriteObjectAny(wrt->GetObject(), OP2TCLASS(wrt), CPyCppyy_PyUnicode_AsString(name),
                                   CPyCppyy_PyUnicode_AsString(option), bufsize);
   } else {
      result = dir->WriteObjectAny(wrt->GetObject(), OP2TCLASS(wrt), CPyCppyy_PyUnicode_AsString(name));
   }
   return PyInt_FromLong((Long_t)result);
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Implement the Get method for TDirectory
/// This allows to seamlessly read from a TDirectory, and therefore a TDirectoryFile
/// and file objects inputting their key name, being them TObjects or not.
PyObject *TDirectoryGet(CPPInstance *self, PyObject *pynamecycle)
{
   // Pythonization of TDirectory::Get that handles non-TObject deriveds
   if (!CPPInstance_Check(self)) {
      PyErr_SetString(PyExc_TypeError, "TDirectory::Get must be called with a TDirectory instance as first argument");
      return nullptr;
   }
   auto dirf = (TDirectory *)OP2TCLASS(self)->DynamicCast(TDirectory::Class(), self->GetObject());
   if (!dirf) {
      PyErr_SetString(PyExc_ReferenceError, "attempt to access a null-pointer");
      return nullptr;
   }
   const char *namecycle = CPyCppyy_PyUnicode_AsString(pynamecycle);
   if (!namecycle)
      return nullptr; // TypeError already set
   auto key = dirf->GetKey(namecycle);
   if (key) {
      void *addr = dirf->GetObjectChecked(namecycle, key->GetClassName());
      return BindCppObjectNoCast(addr, (Cppyy::TCppType_t)Cppyy::GetScope(key->GetClassName()), kFALSE);
   }
   // no key? for better or worse, call normal Get()
   void *addr = dirf->Get(namecycle);
   return BindCppObject(addr, (Cppyy::TCppType_t)Cppyy::GetScope("TObject"), kFALSE);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add attr syntax to TDirectory
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
/// This allows to use TDirectory and daughters (such as TDirectoryFile and TFile)
/// as follows
/// ~~~{.python}
/// myfile.mydir.mysubdir.myHist.Draw()
/// ~~~
PyObject *PyROOT::AddDirectoryAttrSyntaxPyz(PyObject * /* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);
   Utility::AddToClass(pyclass, "__getattr__", (PyCFunction)TDirectoryGet, METH_O);
   Py_RETURN_NONE;
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
