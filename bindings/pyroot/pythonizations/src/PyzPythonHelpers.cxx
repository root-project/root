// Author: Stefan Wunsch CERN  08/2018
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**

Set of helper functions that are invoked from the pythonizors, on the
Python side. For that purpose, they are included in the interface of the
PyROOT extension module.

*/

#include "CPyCppyy.h"
#include "CPPInstance.h"
#include "CPPOverload.h"

#include "PyROOTPythonize.h"

#include "ROOT/RConfig.hxx"
#include "TInterpreter.h"

#include <sstream>

// needed to properly resolve (dllimport) symbols on Windows
namespace CPyCppyy {
   namespace PyStrings {
      R__EXTERN PyObject *gMRO;
   }
}

////////////////////////////////////////////////////////////////////////////
/// \brief Get endianess of the system
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to an empty Python tuple.
/// \return Endianess as Python string
///
/// This function returns endianess of the system as a Python integer. The
/// return value is either '<' or '>' for little or big endian, respectively.
PyObject *PyROOT::GetEndianess(PyObject * /* self */, PyObject * /* args */)
{
#ifdef R__BYTESWAP
   return CPyCppyy_PyText_FromString("<");
#else
   return CPyCppyy_PyText_FromString(">");
#endif
}

using namespace CPyCppyy;

// Helper to add base class methods to the derived class one
static bool AddUsingToClass(PyObject *pyclass, const char *method)
{
   CPPOverload *derivedMethod = (CPPOverload *)PyObject_GetAttrString(pyclass, const_cast<char *>(method));
   if (!CPPOverload_Check(derivedMethod)) {
      Py_XDECREF(derivedMethod);
      return false;
   }

   PyObject *mro = PyObject_GetAttr(pyclass, PyStrings::gMRO);
   if (!mro || !PyTuple_Check(mro)) {
      Py_XDECREF(mro);
      Py_DECREF(derivedMethod);
      return false;
   }

   CPPOverload *baseMethod = nullptr;
   for (int i = 1; i < PyTuple_GET_SIZE(mro); ++i) {
      baseMethod = (CPPOverload *)PyObject_GetAttrString(PyTuple_GET_ITEM(mro, i), const_cast<char *>(method));

      if (!baseMethod) {
         PyErr_Clear();
         continue;
      }

      if (CPPOverload_Check(baseMethod))
         break;

      Py_DECREF(baseMethod);
      baseMethod = nullptr;
   }

   Py_DECREF(mro);

   if (!CPPOverload_Check(baseMethod)) {
      Py_XDECREF(baseMethod);
      Py_DECREF(derivedMethod);
      return false;
   }

   for (PyCallable *pc : baseMethod->fMethodInfo->fMethods) {
      derivedMethod->AdoptMethod(pc->Clone());
   }

   Py_DECREF(baseMethod);
   Py_DECREF(derivedMethod);

   return true;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add base class overloads of a given method to a derived class
/// \param[in] self Always null, since this is a module function.
/// \param[in] args [0] Derived class. [1] Name of the method whose base
/// class overloads to inject in the derived class.
///
/// This function adds base class overloads to a derived class for a given
/// method. This covers the 'using' case, which is not supported by default
/// by the bindings.
PyObject *PyROOT::AddUsingToClass(PyObject * /* self */, PyObject *args)
{
   // Get derived class to pythonize
   PyObject *pyclass = PyTuple_GetItem(args, 0);

   // Get method name where to add overloads
   PyObject *pyname = PyTuple_GetItem(args, 1);
   auto cppname = CPyCppyy_PyText_AsString(pyname);

   AddUsingToClass(pyclass, cppname);

   Py_RETURN_NONE;
}
