// @(#)root/pyroot:$Name:  $:$Id: TPython.cxx,v 1.14 2006/06/14 13:46:30 brun Exp $
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "TPython.h"
#include "ObjectProxy.h"
#include "RootWrapper.h"
#include "TPyClassGenerator.h"

// ROOT
#include "TROOT.h"
#include "TObject.h"

// Standard
#include <stdio.h>
#include <Riostream.h>
#include <string>

//______________________________________________________________________________
//                          Python interpreter access
//                          =========================
//
// The TPython class allows for access to python objects from CINT. The current
// functionality is only basic: ROOT objects and builtin types can freely cross
// the boundary between the two interpreters, python objects can be instantiated
// and their methods can be called. All other cross-coding is based on strings
// that are run on the python interpreter.
//
// Examples:
//
//  $ cat MyPyClass.py
//  print 'creating class MyPyClass ... '
//
//  class MyPyClass:
//     def __init__( self ):
//        print 'in MyPyClass.__init__'
//
//     def gime( self, what ):
//        return what
//
//  $ root -l
//  // Execute a string of python code.
//  root [0] TPython::Exec( "print \'Hello World!\'" );
//  Hello World!
//
//  // Create a TBrowser on the python side, and transfer it back and forth.
//  // Note the required explicit (void*) cast!
//  root [1] TBrowser* b = (void*)TPython::Eval( "ROOT.TBrowser()" );
//  root [2] TPython::Bind( b, "b" );
//  root [3] b == (void*) TPython::Eval( "b" )
//  (int)1
//
//  // Builtin variables can cross-over by using implicit casts.
//  root [4] int i = TPython::Eval( "1 + 1" );
//  root [5] i
//  (int)2
//
//  // Load a python module with a class definition, and use it.
//  root [6] TPython::LoadMacro( "MyPyClass.py" );
//  creating class MyPyClass ...
//  root [7] MyPyClass m;
//  in MyPyClass.__init__
//  root [8] char* s = m.gime( "aap" );
//  root [9] s
//  (char* 0x41ee7754)"aap"
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

   static Bool_t isInitialized = kFALSE;
   if ( isInitialized )
      return kTRUE;

   if ( ! Py_IsInitialized() ) {
   // this happens if CINT comes in first
      PyEval_InitThreads();
      Py_Initialize();

   // try again
      if ( ! Py_IsInitialized() ) {
      // give up ...
         std::cerr << "Error: python has not been intialized; returning." << std::endl;           
         return kFALSE;
      }

   // set argv
      char* argv[] = { const_cast< char* >( "root" ) };
      PySys_SetArgv( sizeof(argv)/sizeof(argv[0]), argv );

   // force loading of ROOT
      PyRun_SimpleString( const_cast< char* >( "import ROOT" ) );
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

// python side class construction, managed by ROOT
   gROOT->AddClassGenerator( new TPyClassGenerator );

// declare success ...
   isInitialized = kTRUE;
   return kTRUE;
}

//____________________________________________________________________________
void TPython::LoadMacro( const char* name )
{
// Execute the give python script as if it were a macro (effectively an
// execfile in __main__), and create CINT equivalents for any newly available
// python classes.

// setup
   if ( ! Initialize() )
      return;

// obtain a reference to look for new classes later
   PyObject* old = PyDict_Values( gMainDict );

// actual execution
   Exec( (std::string( "execfile(\"" ) + name + "\")").c_str() );

// obtain new __main__ contents
   PyObject* current = PyDict_Values( gMainDict );

// create CINT classes for all new python classes
   for ( int i = 0; i < PyList_GET_SIZE( current ); ++i ) {
      PyObject* value = PyList_GET_ITEM( current, i );
      Py_INCREF( value );

      if ( ! PySequence_Contains( old, value ) ) {
      // collect classes
         if ( PyClass_Check( value ) ||
              PyObject_HasAttrString( value, const_cast< char* >( "__bases__" ) ) ) {
         // force class creation
            PyObject* str = PyObject_Str( value );
            gROOT->GetClass( PyString_AS_STRING( str ) );
            Py_DECREF( str );
         }
      }

      Py_DECREF( value );
   }

   Py_DECREF( current );
   Py_DECREF( old );
}

//____________________________________________________________________________
void TPython::ExecScript( const char* name, int argc, const char** argv )
{
// Execute a python stand-alone script, with argv CLI arguments.
//
// example of use:
//    const char* argv[] = { "1", "2", "3" };
//    TPython::ExecScript( "test.py", sizeof(argv)/sizeof(argv[0]), argv );


// setup
   if ( ! Initialize() )
      return;

// verify arguments
   if ( ! name ) {
      std::cerr << "Error: no file name specified." << std::endl;
      return;
   }

   FILE* fp = fopen( name, "r" );
   if ( ! fp ) {
      std::cerr << "Error: could not open file \"" << name << "\"." << std::endl;
      return;
   }

// setup command line part
   PyObject* oldargv = PySys_GetObject( const_cast< char* >( "argv" ) );
   if ( ! oldargv )                               // e.g. apache
      PyErr_Clear();
   else {
      PyObject* l = PyList_New( PyList_GET_SIZE( oldargv ) );
      for ( int i = 1; i < PyList_GET_SIZE( oldargv ); ++i ) {
         PyObject* item = PyList_GET_ITEM( oldargv, i );
         Py_INCREF( item );
         PyList_SET_ITEM( l, i, item );           // steals ref
      }
      oldargv = l;
   }

   argc += 1;
   const char** argv2 = new const char*[ argc ];
   for ( int i = 1; i < argc; ++i ) argv2[ i ] = argv[ i-1 ];
   argv2[ 0 ] = Py_GetProgramName();
   PySys_SetArgv( argc, const_cast< char** >( argv2 ) );
   delete argv2;

// actual script execution
   PyObject* gbl = PyDict_Copy( gMainDict );
   PyObject* result = PyRun_FileEx( fp, name, Py_file_input, gbl, gbl, 1 /* close fp */ );
   if ( ! result )
      PyErr_Print();
   Py_XDECREF( result );
   Py_DECREF( gbl );

// restore original command line
   PySys_SetObject( const_cast< char* >( "argv" ), oldargv );
   Py_XDECREF( oldargv );
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
const TPyReturn TPython::Eval( const char* expr )
{
// Evaluate a python expression (e.g. "ROOT.TBrowser()").
//
// Caution: do not hold on to the return value: either store it in a builtin
// type (implicit casting will work), or in a pointer to a ROOT object (explicit
// casting to a void* is required).

// setup
   if ( ! Initialize() )
      return TPyReturn();

// evaluate the expression
   PyObject* result =
      PyRun_String( const_cast< char* >( expr ), Py_eval_input, gMainDict, gMainDict );

// report errors as appropriate; return void
   if ( ! result ) {
      PyErr_Print();
      return TPyReturn();
   }

// results that require no converion
   if ( result == Py_None || PyROOT::ObjectProxy_Check( result ) )
      return TPyReturn( result );

// explicit conversion for python type required
   PyObject* pyclass = PyObject_GetAttr( result, gClassString );
   if ( pyclass != 0 ) {
   // retrieve class name and the module in which it resides
      PyObject* name = PyObject_GetAttr( pyclass, gNameString );
      PyObject* module = PyObject_GetAttr( pyclass, gModuleString );

   // concat name
      std::string qname =
         std::string( PyString_AS_STRING( module ) ) + '.' + PyString_AS_STRING( name );
      Py_DECREF( module );
      Py_DECREF( name );
      Py_DECREF( pyclass );

   // locate ROOT style class with this name
      TClass* klass = gROOT->GetClass( qname.c_str() );

   // construct general ROOT python object that pretents to be of class 'klass'
      if ( klass != 0 )
         return TPyReturn( result );
   } else
      PyErr_Clear();

// no conversion, return null pointer object
   Py_DECREF( result );
   return TPyReturn();
}

//____________________________________________________________________________
Bool_t TPython::Bind( TObject* object, const char* label )
{
// Bind a ROOT object with, at the python side, the name "label".

// check given address and setup
   if ( ! ( object && Initialize() ) )
      return kFALSE;

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

   return kFALSE;
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
