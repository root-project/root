// Author: Danilo Piparo CERN  08/2018
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
#include "CallContext.h"
#include "PyROOTPythonize.h"
#include "TPython.h"
#include "Utility.h"

#include "PyzCppHelpers.hxx"

#include "TFile.h"

#include "Python.h"

using namespace CPyCppyy;

PyObject* TFileGetAttr( PyObject* self, PyObject* attr )
{
// Pythonization of TFile::Get that raises AttributeError on failure.
   PyObject* result = CallPyObjMethod( self, "Get", attr );
   if ( !result ) {
      return result;
   }

   if ( !PyObject_IsTrue( result ) ) {
      PyObject* astr = PyObject_Str( attr );
      PyErr_Format( PyExc_AttributeError, "TFile object has no attribute \'%s\'",
                     CPyCppyy_PyUnicode_AsString( astr ) );
      Py_DECREF( astr );
      Py_DECREF( result );
      return nullptr;
   }

   // caching behavior seems to be more clear to the user; can always override said
   // behavior (i.e. re-read from file) with an explicit Get() call
      PyObject_SetAttr( self, attr, result );
      return result;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add pythonizations to the File class.
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
PyObject *PyROOT::PythonizeTFile(PyObject */* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);

   // TFile::Open really is a constructor, really
   PyObject* attr = PyObject_GetAttrString( pyclass, (char*)"Open" );
   if ( TPython::CPPOverload_Check( attr ) ) {
      ((CPPOverload*)attr)->fMethodInfo->fFlags |= CallContext::kIsCreator;
   }
   Py_XDECREF( attr );

   // allow member-style access to entries in file
   Utility::AddToClass( pyclass, "__getattr__", (PyCFunction) TFileGetAttr, METH_O );

   Py_RETURN_NONE;
}

