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
#include "TPython.h"
#include "Utility.h"

#include "PyzCppHelpers.hxx"

#include "TClass.h"
#include "TDirectory.h"

#include "Python.h"

using namespace CPyCppyy;

PyObject* TDirectoryGetObject( CPPInstance* self, PyObject* args )
{
   PyObject* name = nullptr;
   CPPInstance* ptr = nullptr;
   if ( ! PyArg_ParseTuple( args, const_cast< char* >( "O!O!:TDirectory::GetObject" ),
            &CPyCppyy_PyUnicode_Type, &name, &CPPInstance_Type, &ptr ) )
      return nullptr;

   auto dir =
      (TDirectory*)OP2TCLASS(self)->DynamicCast( TDirectory::Class(), self->GetObject() );

   if ( ! dir ) {
      PyErr_SetString( PyExc_TypeError,
         "TDirectory::GetObject must be called with a TDirectory instance as first argument" );
      return nullptr;
   }

   void* address = dir->GetObjectChecked( CPyCppyy_PyUnicode_AsString( name ), OP2TCLASS(ptr) );
   if ( address ) {
      ptr->Set( address );

      Py_INCREF( Py_None );
      return Py_None;
   }

   PyErr_Format( PyExc_LookupError, "no such object, \"%s\"", CPyCppyy_PyUnicode_AsString( name ) );
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add pythonizations to the TDirectory class.
/// \param[in] self Always null, since this is a module function.
/// \param[in] args Pointer to a Python tuple object containing the arguments
PyObject *PyROOT::PythonizeTDirectory(PyObject */* self */, PyObject *args)
{
   PyObject *pyclass = PyTuple_GetItem(args, 0);

   Utility::AddToClass( pyclass, "GetObject", (PyCFunction) TDirectoryGetObject );

   Py_RETURN_NONE;
}

