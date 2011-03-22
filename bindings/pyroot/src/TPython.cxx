// @(#)root/pyroot:$Id$
// Author: Wim Lavrijsen, Apr 2004

// Bindings
#include "PyROOT.h"
#include "PyStrings.h"
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
//  // Load a python module with a class definition, and use it. Casts are
//  // necessary as the type information can not be otherwise derived.
//  root [6] TPython::LoadMacro( "MyPyClass.py" );
//  creating class MyPyClass ...
//  root [7] MyPyClass m;
//  in MyPyClass.__init__
//  root [8] std::string s = (char*)m.gime( "aap" );
//  root [9] s
//  (class TString)"aap"
//
// It is possible to switch between interpreters by calling "TPython::Prompt()"
// on the CINT side, while returning with ^D (EOF). State is preserved between
// successive switches.
//
// The API part provides (direct) C++ access to the bindings functionality of
// PyROOT. It allows verifying that you deal with a PyROOT python object in the
// first place (ObjectProxy_Check for ObjectProxy and any derived types, as well
// as ObjectProxy_CheckExact for ObjectProxy's only); and it allows conversions
// of void* to an ObjectProxy and vice versa.


//- data ---------------------------------------------------------------------
ClassImp(TPython)
static PyObject* gMainDict = 0;


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

   // try again to initialize the interpreter
      if ( ! Py_IsInitialized() ) {
      // give up ...
         std::cerr << "Error: python has not been intialized; returning." << std::endl;           
         return kFALSE;
      }

   // set the command line arguments on python's sys.argv
#if PY_VERSION_HEX < 0x03000000
      char* argv[] = { const_cast< char* >( "root" ) };
#else
      wchar_t* argv[] = { const_cast< wchar_t* >( L"root" ) };
#endif
      PySys_SetArgv( sizeof(argv)/sizeof(argv[0]), argv );

   // force loading of the ROOT module
      PyRun_SimpleString( const_cast< char* >( "import ROOT" ) );
   }

   if ( ! gMainDict ) {
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
         if ( PyClass_Check( value ) || PyObject_HasAttr( value, PyROOT::PyStrings::gBases ) ) {
         // get full class name (including module)
            PyObject* pyModName = PyObject_GetAttr( value, PyROOT::PyStrings::gModule );
            PyObject* pyClName  = PyObject_GetAttr( value, PyROOT::PyStrings::gName );

            if ( PyErr_Occurred() )
               PyErr_Clear();

         // need to check for both exact and derived (differences exist between older and newer
         // versions of python ... bug?)
            if ( (pyModName && pyClName) &&\
                 ( (PyBytes_CheckExact( pyModName ) && PyBytes_CheckExact( pyClName )) ||\
                   (PyBytes_Check( pyModName ) && PyBytes_Check( pyClName ))\
                 ) ) {
            // build full, qualified name
               std::string fullname = PyROOT_PyUnicode_AsString( pyModName );
               fullname += '.';
               fullname += PyROOT_PyUnicode_AsString( pyClName );

            // force class creation (this will eventually call TPyClassGenerator)
               TClass::GetClass( fullname.c_str(), kTRUE );
            }

            Py_XDECREF( pyClName );
            Py_XDECREF( pyModName );
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

// store a copy of the old cli for restoration
   PyObject* oldargv = PySys_GetObject( const_cast< char* >( "argv" ) );   // borrowed
   if ( ! oldargv )                               // e.g. apache
      PyErr_Clear();
   else {
      PyObject* l = PyList_New( PyList_GET_SIZE( oldargv ) );
      for ( int i = 0; i < PyList_GET_SIZE( oldargv ); ++i ) {
         PyObject* item = PyList_GET_ITEM( oldargv, i );
         Py_INCREF( item );
         PyList_SET_ITEM( l, i, item );           // steals ref
      }
      oldargv = l;
   }

// create and set (add progam name) the new command line
   argc += 1;
#if PY_VERSION_HEX < 0x03000000
   const char** argv2 = new const char*[ argc ];
   for ( int i = 1; i < argc; ++i ) argv2[ i ] = argv[ i-1 ];
   argv2[ 0 ] = Py_GetProgramName();
   PySys_SetArgv( argc, const_cast< char** >( argv2 ) );
   delete [] argv2;
#else
// TODO: fix this to work like above ...
   argv = 0;
#endif

// actual script execution
   PyObject* gbl = PyDict_Copy( gMainDict );
   PyObject* result =   // PyRun_FileEx closes fp (b/c of last argument "1")
      PyRun_FileEx( fp, const_cast< char* >( name ), Py_file_input, gbl, gbl, 1 );
   if ( ! result )
      PyErr_Print();
   Py_XDECREF( result );
   Py_DECREF( gbl );

// restore original command line
   if ( oldargv ) {
      PySys_SetObject( const_cast< char* >( "argv" ), oldargv );
      Py_DECREF( oldargv );
   }
}

//____________________________________________________________________________
Bool_t TPython::Exec( const char* cmd )
{
// Execute a python statement (e.g. "import ROOT").

// setup
   if ( ! Initialize() )
      return kFALSE;

// execute the command
   PyObject* result =
      PyRun_String( const_cast< char* >( cmd ), Py_file_input, gMainDict, gMainDict );

// test for error
   if ( result ) {
      Py_DECREF( result );
      return kTRUE;
   } else {
      PyErr_Print();
      return kFALSE;
   }
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

// results that require no convserion
   if ( result == Py_None || PyROOT::ObjectProxy_Check( result ) ||
         PyBytes_Check( result ) ||
         PyFloat_Check( result ) || PyLong_Check( result ) || PyInt_Check( result ) )
      return TPyReturn( result );

// explicit conversion for python type required
   PyObject* pyclass = PyObject_GetAttr( result, PyROOT::PyStrings::gClass );
   if ( pyclass != 0 ) {
   // retrieve class name and the module in which it resides
      PyObject* name = PyObject_GetAttr( pyclass, PyROOT::PyStrings::gName );
      PyObject* module = PyObject_GetAttr( pyclass, PyROOT::PyStrings::gModule );

   // concat name
      std::string qname =
         std::string( PyROOT_PyUnicode_AsString( module ) ) + '.' + PyROOT_PyUnicode_AsString( name );
      Py_DECREF( module );
      Py_DECREF( name );
      Py_DECREF( pyclass );

   // locate ROOT style class with this name
      TClass* klass = TClass::GetClass( qname.c_str() );

   // construct general ROOT python object that pretends to be of class 'klass'
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

//____________________________________________________________________________
Bool_t TPython::ObjectProxy_Check( PyObject* pyobject )
{
// Test whether the type of the given pyobject is of ObjectProxy type or any
// derived type.

// setup
   if ( ! Initialize() )
      return kFALSE;

// detailed walk through inheritance hierarchy
   return PyROOT::ObjectProxy_Check( pyobject );
}

//____________________________________________________________________________
Bool_t TPython::ObjectProxy_CheckExact( PyObject* pyobject )
{
// Test whether the type of the given pyobject is ObjectProxy type.

// setup
   if ( ! Initialize() )
      return kFALSE;

// direct pointer comparison of type member
   return PyROOT::ObjectProxy_CheckExact( pyobject );
}

//____________________________________________________________________________
void* TPython::ObjectProxy_AsVoidPtr( PyObject* pyobject )
{
// Extract the object pointer held by the ObjectProxy pyobject.

// setup
   if ( ! Initialize() )
      return 0;

// check validity of cast
   if ( ! PyROOT::ObjectProxy_Check( pyobject ) )
      return 0;

// get held object (may be null)
   return ((PyROOT::ObjectProxy*)pyobject)->GetObject();
}

//____________________________________________________________________________
PyObject* TPython::ObjectProxy_FromVoidPtr( void* addr, const char* classname )
{
// Bind the addr to a python object of class defined by classname.

// setup
   if ( ! Initialize() )
      return 0;

// perform cast (the call will check TClass and addr, and set python errors)
   return PyROOT::BindRootObjectNoCast( addr, TClass::GetClass( classname ), kFALSE );
}
