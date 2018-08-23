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
#include "CPPInstance.h"
#include "CPPOverload.h"
#include "CallContext.h"
#include "Converters.h"
#include "ProxyWrappers.h"
#include "PyROOTPythonize.h"
#include "TClass.h"
#include "TKey.h"
#include "TPython.h"
#include "Utility.h"

#include "TFile.h"

#include "Python.h"



// FIXME: Duplicate from Pythonize.cxx: see ROOT-9510
inline PyObject* CallPyObjMethod(PyObject* obj, const char* meth, PyObject* arg1)
{
// Helper; call method with signature: obj->meth(arg1).
    Py_INCREF(obj);
    PyObject* result = PyObject_CallMethod(
        obj, const_cast<char*>(meth), const_cast<char*>("O"), arg1);
    Py_DECREF(obj);
    return result;
}

using namespace CPyCppyy;

// FIXME: This method should be in some sort of external helper file
static inline TClass* OP2TCLASS( CPPInstance* pyobj ) {
   return TClass::GetClass( Cppyy::GetFinalName( pyobj->ObjectIsA() ).c_str());
}

// This is done for TFile, but Get() is really defined in TDirectoryFile and its base
// TDirectory suffers from a similar problem. Nevertheless, the TFile case is by far
// the most common, so we'll leave it at this until someone asks for one of the bases
// to be pythonized.
PyObject* TDirectoryFileGet( CPPInstance* self, PyObject* pynamecycle )
{
// Pythonization of TDirectoryFile::Get that handles non-TObject deriveds
   if ( ! CPPInstance_Check( self ) ) {
      PyErr_SetString( PyExc_TypeError,
         "TDirectoryFile::Get must be called with a TDirectoryFile instance as first argument" );
      return nullptr;
   }

   auto dirf =
      (TDirectoryFile*)OP2TCLASS(self)->DynamicCast( TDirectoryFile::Class(), self->GetObject() );
   if ( !dirf ) {
      PyErr_SetString( PyExc_ReferenceError, "attempt to access a null-pointer" );
      return nullptr;
   }

   const char* namecycle = CPyCppyy_PyUnicode_AsString( pynamecycle );
   if ( !namecycle )
      return nullptr;     // TypeError already set

   auto key = dirf->GetKey( namecycle );
   if ( key ) {
      void* addr = dirf->GetObjectChecked( namecycle, key->GetClassName() );
      return BindCppObjectNoCast( addr,
         (Cppyy::TCppType_t)Cppyy::GetScope( key->GetClassName() ), kFALSE );
   }

   // no key? for better or worse, call normal Get()
   void* addr = dirf->Get( namecycle );
   return BindCppObject( addr, (Cppyy::TCppType_t)Cppyy::GetScope( "TObject" ), kFALSE );
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add pythonizations to the TDirectoryFile class.
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
PyObject *PyROOT::PythonizeTDirectoryFile(PyObject */* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);

   Utility::AddToClass( pyclass, "Get", (PyCFunction) TDirectoryFileGet, METH_O );

   Py_RETURN_NONE;
}

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

