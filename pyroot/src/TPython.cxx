// @(#)root/pyroot:$Name:  $:$Id:  $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "RootWrapper.h"
#include "ObjectHolder.h"
#include "Utility.h"
#include "TPython.h"

// ROOT
#include "TROOT.h"
#include "TObject.h"

// Standard
#include <stdio.h>
#include <iostream>


//- data ________________________________________________________________________
namespace {

   PyObject* g_maindict = 0;

} // unnamed namespace


//- private helpers -------------------------------------------------------------
bool Python::initialize_() {
   if ( ! Py_IsInitialized() ) {
   // this happens if CINT comes in first
      PyEval_InitThreads();
      Py_Initialize();

   // try again
      if ( ! Py_IsInitialized() ) {
      // give up ...
         std::cout << "Error: python has not been intialized; returning." << std::endl;           
         return false;
      }

   // set argv
      char* argv[] = { const_cast< char* >( "root" ) };
      PySys_SetArgv( sizeof(argv)/sizeof(argv[0]), argv );

   // force loading of ROOT
      PyRun_SimpleString( const_cast< char* >( "import ROOT" ) );
   }

   if ( g_maindict == 0 ) {
   // retrieve the main dictionary
      g_maindict = PyModule_GetDict(
         PyImport_AddModule( const_cast< char* >( "__main__" ) ) );
      Py_INCREF( g_maindict );
   }

// declare success ...
   return true;
}


//- CINT entry points -----------------------------------------------------------
void Python::exec( char* cmd ) {
// setup
   if ( ! initialize_() )
      return;

// execute the command
   PyObject* result = PyRun_String( cmd, Py_file_input, g_maindict, g_maindict );

// test for error
   if ( result )
      Py_DECREF( result );
   else
      PyErr_Print();
}


TObject* Python::eval( char* expr ) {
// setup
   if ( ! initialize_() )
      return 0;

// evaluate the expression
   PyObject* result = PyRun_String( expr, Py_eval_input, g_maindict, g_maindict );

// test for error
   if ( ! result ) {
      PyErr_Print();
      return 0;
   }

// test for a usuable result
   if ( result == Py_None ) {
      Py_DECREF( result );
      return 0;
   }

// the result is a new handle, for testing use a borrowed, since ROOT will have to take
// care of the clean up (need to figure out how)
   PyROOT::ObjectHolder* holder = PyROOT::Utility::getObjectHolder( result );
   if ( holder != 0 )
      return reinterpret_cast< TObject* >( holder->getObject() );

   Py_DECREF( result );
   return 0;
}


bool Python::bind( TObject* obj, char* label ) {
// check given address and setup
   if ( ! ( obj && initialize_() ) )
      return false;

// bind object in the main namespace
   TClass* cls = obj->IsA();
   if ( cls != 0 ) {
      PyObject* bound =
         PyROOT::bindRootObject( new PyROOT::ObjectHolder( (void*)obj, cls, false ) );

      if ( bound ) {
         bool bOk = PyDict_SetItemString( g_maindict, label, bound ) == 0;
         Py_DECREF( bound );

         return bOk;
      }
   }

   return false;
}


void Python::prompt() {
// setup
   if ( ! initialize_() ) {
      return;
   }

// enter i/o interactive mode
   PyRun_InteractiveLoop( stdin, const_cast< char* >( "\0" ) );
}
