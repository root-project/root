// @(#)root/pyroot:$Name:  $:$Id: TPython.cxx,v 1.2 2004/04/27 14:44:02 rdm Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "RootWrapper.h"
#include "ObjectHolder.h"
#include "Utility.h"
#include "TPython.h"
#include "TPyReturn.h"

// ROOT
#include "TROOT.h"
#include "TObject.h"

// Standard
#include <stdio.h>
#include <Riostream.h>


namespace {

   PyObject* g_maindict = 0;

   PyObject* g_pystr_class  = 0;
   PyObject* g_pystr_name   = 0;
   PyObject* g_pystr_module = 0;

} // unnamed namespace


bool TPython::Initialize()  {
// Private initialization method.

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
   // initialize some handy strings
      g_pystr_class  = PyString_FromString( "__class__" );
      g_pystr_name   = PyString_FromString( "__name__" );
      g_pystr_module = PyString_FromString( "__module__" );

   // retrieve the main dictionary
      g_maindict = PyModule_GetDict(
         PyImport_AddModule( const_cast< char* >( "__main__" ) ) );
      Py_INCREF( g_maindict );
   }

// declare success ...
   return true;
}


void TPython::Exec( const char* cmd ) {
// Execute a python statement (e.g. "import ROOT").

// setup
   if ( ! Initialize() )
      return;

// execute the command
   PyObject* result =
      PyRun_String( const_cast< char* >( cmd ), Py_file_input, g_maindict, g_maindict );

// test for error
   if ( result )
      Py_DECREF( result );
   else
      PyErr_Print();
}


const TPyReturn& TPython::Eval( const char* expr ) {
// Evaluate a python expression (e.g. "ROOT.TBrowser()"). TODO: fix memory management

// setup
   if ( ! Initialize() )
      return *(new TPyReturn( 0, 0 ));

// evaluate the expression
   PyObject* result =
      PyRun_String( const_cast< char* >( expr ), Py_eval_input, g_maindict, g_maindict );

// test for error
   if ( ! result ) {
      PyErr_Print();
      return *(new TPyReturn( 0, 0 ));
   }

// test for a usuable result
   if ( result == Py_None ) {
      Py_DECREF( result );
      return *(new TPyReturn( 0, 0 ));
   }

// let the TObject& fill the place of the PyObject&
   PyROOT::ObjectHolder* holder = PyROOT::Utility::getObjectHolder( result );
   if ( holder != 0 ) {
      TObject* obj = (TObject*)holder->getObject();
      if ( obj != 0 ) {
         TObject& robj = *obj;
         return static_cast< TPyReturn& >( robj );
      }
      else
         return *(new TPyReturn( 0, 0 ));
   }

// apparently no ROOT object, try to convert from python
   PyObject* pyclass = PyObject_GetAttr( result, g_pystr_class );
   if ( pyclass != 0 ) {
   // retrieve class name and the module in which it resides
      PyObject* name = PyObject_GetAttr( pyclass, g_pystr_name );
      PyObject* module = PyObject_GetAttr( pyclass, g_pystr_module );

   // concatename
      std::string qname =
         std::string( PyString_AS_STRING( module ) ) + '.' + PyString_AS_STRING( name );
      Py_DECREF( module );
      Py_DECREF( name );
      Py_DECREF( pyclass );

   // locate ROOT style class with this name
      TClass* cls = gROOT->GetClass( qname.c_str() );

   // construct general ROOT python object that pretents to be of class 'cls'
      if ( cls != 0 ) {
         return *(new TPyReturn( result, cls ));  // steals ref to 'result'
      }
   }
   else {
      PyErr_Clear();
   }

// no conversion, return null pointer object
   Py_DECREF( result );
   return *(new TPyReturn( 0, 0 ));
}


bool TPython::Bind( TObject* obj, const char* label ) {
// Bind a ROOT object with, at the python side, the name "label".

// check given address and setup
   if ( ! ( obj && Initialize() ) )
      return false;

// bind object in the main namespace
   TClass* cls = obj->IsA();
   if ( cls != 0 ) {
      PyObject* bound =
         PyROOT::bindRootObject( new PyROOT::ObjectHolder( (void*)obj, cls, false ) );

      if ( bound ) {
         bool bOk = PyDict_SetItemString( g_maindict, const_cast< char* >( label ), bound ) == 0;
         Py_DECREF( bound );

         return bOk;
      }
   }

   return false;
}


void TPython::Prompt() {
// Enter an interactive python session (exit with ^D).

// setup
   if ( ! Initialize() ) {
      return;
   }

// enter i/o interactive mode
   PyRun_InteractiveLoop( stdin, const_cast< char* >( "\0" ) );
}
