// @(#)root/pyroot:$Name:  $:$Id: TPython.cxx,v 1.1 2004/04/27 06:28:48 brun Exp $
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
#include <Riostream.h>


namespace {

   PyObject* g_maindict = 0;

} // unnamed namespace


//______________________________________________________________________________
bool TPython::Initialize() 
{
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
      // retrieve the main dictionary
      g_maindict = PyModule_GetDict(
         PyImport_AddModule( const_cast< char* >( "__main__" ) ) );
      Py_INCREF( g_maindict );
   }

   // declare success ...
   return true;
}

//______________________________________________________________________________
void TPython::Exec(const char *cmd)
{
   // Execute a python statement (e.g. "import ROOT").

   // setup
   if ( ! Initialize() )
      return;

   // execute the command
   PyObject* result = PyRun_String( const_cast< char * >(cmd), Py_file_input, g_maindict, g_maindict );

   // test for error
   if ( result )
      Py_DECREF( result );
   else
      PyErr_Print();
}

//______________________________________________________________________________
TObject *TPython::Eval(const char* expr)
{
   // Evaluate a python expression (e.g. "1+1").

   // setup
   if ( ! Initialize() )
      return 0;

   // evaluate the expression
   PyObject* result = PyRun_String( const_cast< char * >(expr), Py_eval_input, g_maindict, g_maindict );

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

   // the result is a new handle, for testing use a borrowed, since ROOT
   // will have to take care of the clean up (need to figure out how)
   PyROOT::ObjectHolder* holder = PyROOT::Utility::getObjectHolder( result );
   if ( holder != 0 )
      return reinterpret_cast< TObject* >( holder->getObject() );

   Py_DECREF( result );
   return 0;
}

//______________________________________________________________________________
bool TPython::Bind(TObject *obj, const char *label)
{
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
         bool bOk = PyDict_SetItemString( g_maindict, const_cast< char * >(label), bound ) == 0;
         Py_DECREF( bound );

         return bOk;
      }
   }

   return false;
}

//______________________________________________________________________________
void TPython::Prompt()
{
   // Enter an interactive python session (exit with ^D).

   // setup
   if ( ! Initialize() ) {
      return;
   }

   // enter i/o interactive mode
   PyRun_InteractiveLoop( stdin, const_cast< char* >( "\0" ) );
}
