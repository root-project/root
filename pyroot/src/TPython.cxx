// @(#)root/pyroot:$Name:  $:$Id: TPython.cxx,v 1.6 2004/11/23 21:45:06 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "TPython.h"
#include "ObjectProxy.h"
#include "RootWrapper.h"

// ROOT
#include "TROOT.h"
#include "TObject.h"

// Standard
#include <stdio.h>
#include <Riostream.h>

//______________________________________________________________________________
//                          Python interpreter access
//                          =========================
//
// The TPython class allows for access to python objects from CINT. The current
// functionality is only basic: ROOT objects and builtin types can cross the
// boundary between the two interpreters. All other cross-coding is based on
// strings that are run on the python interpreter.
//
// Example: Accessing the python interpreter from ROOT
//
// root [0] gSystem->Load( "libPyROOT" );
// (int)0
// root [1] TPython::Exec( "print 1 + 1" );  // write '2' to stdout
// 2
//
// // create a TBrowser on the python side, and transfer it back and forth
// root [2] TBrowser* b = (TBrowser*) Python::Eval( "ROOT.TBrowser()" );
// root [3] TPython::Bind( b, "b" );
// root [4] b == (TBrowser*) TPython::Eval( "b" )
// (int)1
//
// // builtin variables can cross-over
// root [5] int i = TPython::Eval( "1 + 1" );
// root [6] i
// (int)2
//
// It is possible to switch between interpreters by calling "TPython::Prompt()"
// on the CINT side, while returning with ^D (EOF). State is preserved between
// successive switches.


//- data ---------------------------------------------------------------------
ClassImp(TPython)

namespace {

   PyObject* gMainDict = 0;

   PyObject* gClassString  = 0;
   PyObject* gNameString   = 0;
   PyObject* gModuleString = 0;

} // unnamed namespace


//- static public members ----------------------------------------------------
Bool_t TPython::Initialize()
{
// Private initialization method: setup the python interpreter and load the
// ROOT module.

   if ( ! Py_IsInitialized() ) {
   // this happens if CINT comes in first
      PyEval_InitThreads();
      Py_Initialize();

   // try again
      if ( ! Py_IsInitialized() ) {
      // give up ...
         std::cerr << "Error: python has not been intialized; returning." << std::endl;           
         return false;
      }

   // set argv
      char* argv[] = { const_cast< char* >( "root" ) };
      PySys_SetArgv( sizeof(argv)/sizeof(argv[0]), argv );

   // force loading of ROOT
      PyRun_SimpleString( const_cast< char* >( "from ROOT import *" ) );
   }

   if ( ! gMainDict ) {
   // initialize some handy strings
      gClassString  = PyString_FromString( "__class__" );
      gNameString   = PyString_FromString( "__name__" );
      gModuleString = PyString_FromString( "__module__" );

   // retrieve the main dictionary
      gMainDict = PyModule_GetDict(
         PyImport_AddModule( const_cast< char* >( "__main__" ) ) );
      Py_INCREF( gMainDict );
   }

// declare success ...
   return true;
}

//____________________________________________________________________________
void TPython::Exec( const char* cmd )
{
// Execute a python statement (e.g. "import ROOT").

// setup
   if ( ! Initialize() )
      return;

// execute the command
   PyObject* result =
      PyRun_String( const_cast< char* >( cmd ), Py_file_input, gMainDict, gMainDict );

// test for error
   if ( result )
      Py_DECREF( result );
   else
      PyErr_Print();
}


//____________________________________________________________________________
const TPyReturn& TPython::Eval( const char* expr )
{
// Evaluate a python expression (e.g. "ROOT.TBrowser()").
//
// Caution: do not hold on to the return value: either store it in a builtin
// type (implicit casting will work), or in a pointer to a ROOT object (explicit
// casting is required).

// setup
   if ( ! Initialize() )
      return *(new TPyReturn( 0, 0 ));

// evaluate the expression
   PyObject* result =
      PyRun_String( const_cast< char* >( expr ), Py_eval_input, gMainDict, gMainDict );

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
   if ( PyROOT::ObjectProxy_Check( result ) ) {
      TObject* obj = (TObject*)((PyROOT::ObjectProxy*)result)->GetObject();
      if ( obj != 0 ) {
         TObject& robj = *obj;
         return static_cast< TPyReturn& >( robj );
      } else
         return *(new TPyReturn( 0, 0 ));
   }

// apparently no ROOT object, try to convert from python
   PyObject* pyclass = PyObject_GetAttr( result, gClassString );
   if ( pyclass != 0 ) {
   // retrieve class name and the module in which it resides
      PyObject* name = PyObject_GetAttr( pyclass, gNameString );
      PyObject* module = PyObject_GetAttr( pyclass, gModuleString );

   // concatename
      std::string qname =
         std::string( PyString_AS_STRING( module ) ) + '.' + PyString_AS_STRING( name );
      Py_DECREF( module );
      Py_DECREF( name );
      Py_DECREF( pyclass );

   // locate ROOT style class with this name
      TClass* klass = gROOT->GetClass( qname.c_str() );

   // construct general ROOT python object that pretents to be of class 'klass'
      if ( klass != 0 ) {
         return *(new TPyReturn( result, klass ));  // steals ref to 'result'
      }
   } else
      PyErr_Clear();

// no conversion, return null pointer object
   Py_DECREF( result );
   return *(new TPyReturn( 0, 0 ));
}

//____________________________________________________________________________
Bool_t TPython::Bind( TObject* object, const char* label )
{
// Bind a ROOT object with, at the python side, the name "label".

// check given address and setup
   if ( ! ( object && Initialize() ) )
      return false;

// bind object in the main namespace
   TClass* klass = object->IsA();
   if ( klass != 0 ) {
      PyObject* bound = PyROOT::BindRootObject( (void*)object, klass );

      if ( bound ) {
         Bool_t bOk = PyDict_SetItemString( gMainDict, const_cast< char* >( label ), bound ) == 0;
         Py_DECREF( bound );

         return bOk;
      }
   }

   return false;
}

//____________________________________________________________________________
void TPython::Prompt() {
// Enter an interactive python session (exit with ^D). State is preserved
// between successive calls.

// setup
   if ( ! Initialize() ) {
      return;
   }

// enter i/o interactive mode
   PyRun_InteractiveLoop( stdin, const_cast< char* >( "\0" ) );
}
