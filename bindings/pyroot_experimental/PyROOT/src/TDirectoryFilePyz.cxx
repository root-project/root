// Author: Danilo Piparo CERN  08/2018
// Original PyROOT code by Wim Lavrijsen, LBL

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
#include "ProxyWrappers.h"
#include "PyROOTPythonize.h"
#include "Utility.h"

#include "TDirectoryFile.h"
#include "TKey.h"

#include "TPython.h"

using namespace CPyCppyy;

// This is done for TFile, but Get() is really defined in TDirectoryFile and its base
// TDirectory suffers from a similar problem. Nevertheless, the TFile case is by far
// the most common, so we'll leave it at this until someone asks for one of the bases
// to be pythonized.
PyObject *TDirectoryFileGet(CPPInstance *self, PyObject *pynamecycle)
{
   // Pythonization of TDirectoryFile::Get that handles non-TObject deriveds
   if (!CPPInstance_Check(self)) {
      PyErr_SetString(PyExc_TypeError,
                      "TDirectoryFile::Get must be called with a TDirectoryFile instance as first argument");
      return nullptr;
   }

   auto dirf = (TDirectoryFile *)OP2TCLASS(self)->DynamicCast(TDirectoryFile::Class(), self->GetObject());
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
/// \brief Add pythonizations to the TDirectoryFile class.
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
PyObject *PyROOT::PythonizeTDirectoryFile(PyObject * /* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);

   Utility::AddToClass(pyclass, "Get", (PyCFunction)TDirectoryFileGet, METH_O);

   Py_RETURN_NONE;
}
