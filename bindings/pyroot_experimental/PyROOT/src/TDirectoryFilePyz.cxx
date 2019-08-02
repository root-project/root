// Author: Massimiliano Galli CERN  08/2019
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "CPyCppyy.h"
#include "CPPOverload.h"
#include "PyROOTPythonize.h"
#include "Python.h"
#include "Utility.h"
#include "PyzCppHelpers.hxx"

using namespace CPyCppyy;

/// Allow access to objects as if they were data members
/// This concerns both TDirectoryFile and TFile, since the latter
/// inherits the Get method from the former
PyObject *TDirectoryFileGetPyz(PyObject *self, PyObject *attr)
{
   // Pythonization of TDirectoryFile::Get that raises AttributeError on failure.
   PyObject *result = CallPyObjMethod(self, "Get", attr);
   if (!result)
      return result;

   if (!PyObject_IsTrue(result)) {
      PyObject *astr = PyObject_Str(attr);
      PyErr_Format(PyExc_AttributeError, "TFile object has no attribute \'%s\'", CPyCppyy_PyUnicode_AsString(astr));
      Py_DECREF(astr);
      Py_DECREF(result);
      return nullptr;
   }

   // caching behavior seems to be more clear to the user; can always override said
   // behavior (i.e. re-read from file) with an explicit Get() call
   PyObject_SetAttr(self, attr, result);
   return result;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Allow access to objects as if they were data member
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
/// received from Python.
///
/// Allow TDirectoryFile and TFile instantiations to access objects as if they
/// were data member; the object retrieved is not a copy but the actual object
/// (e.g. dirfile.object)
PyObject *PyROOT::AddTDirectoryFileGetPyz(PyObject * /* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);
   Utility::AddToClass(pyclass, "__getattr__", (PyCFunction)TDirectoryFileGetPyz, METH_O);
   Py_RETURN_NONE;
}
