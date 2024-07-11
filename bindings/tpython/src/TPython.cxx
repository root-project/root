// Author: Enric Tejedor CERN  08/2019
// Original PyROOT code by Wim Lavrijsen, LBL
//
// /*************************************************************************
//  * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
//  * All rights reserved.                                                  *
//  *                                                                       *
//  * For the licensing terms see $ROOTSYS/LICENSE.                         *
//  * For the list of contributors see $ROOTSYS/README/CREDITS.             *
//  *************************************************************************/

// Bindings
// CPyCppyy.h must be go first, since it includes Python.h, which must be
// included before any standard header
#include "CPyCppyy/API.h"
#include "TPython.h"

// ROOT
#include "TROOT.h"
#include "TClassRef.h"
#include "TObject.h"

// Standard
#include <stdio.h>
#include <Riostream.h>
#include <string>

/// \class TPython
/// Accessing the Python interpreter from C++.
///
/// The TPython class allows for access to python objects from Cling. The current
/// functionality is only basic: ROOT objects and builtin types can freely cross
/// the boundary between the two interpreters, python objects can be instantiated
/// and their methods can be called. All other cross-coding is based on strings
/// that are run on the python interpreter.
///
/// Examples:
///
/// ~~~{.cpp}
///  $ root -l
///  // Execute a string of python code.
///  root [0] TPython::Exec( "print(\'Hello World!\')" );
///  Hello World!
///
///  // Create a TBrowser on the python side, and transfer it back and forth.
///  // Note the required explicit (void*) cast!
///  root [1] TBrowser* b = (void*)TPython::Eval( "ROOT.TBrowser()" );
///  root [2] TPython::Bind( b, "b" );
///  root [3] b == (void*) TPython::Eval( "b" )
///  (int)1
///
///  // Builtin variables can cross-over by using implicit casts.
///  root [4] int i = TPython::Eval( "1 + 1" );
///  root [5] i
///  (int)2
/// ~~~
///
/// And with a python file `MyPyClass.py` like this:
/// ~~~{.py}
///  print 'creating class MyPyClass ... '
///
///  class MyPyClass:
///     def __init__( self ):
///        print 'in MyPyClass.__init__'
///
///     def gime( self, what ):
///        return what
/// ~~~
/// one can load a python module, and use the class. Casts are
/// necessary as the type information can not be otherwise derived.
/// ~~~{.cpp}
///  root [6] TPython::LoadMacro( "MyPyClass.py" );
///  creating class MyPyClass ...
///  root [7] MyPyClass m;
///  in MyPyClass.__init__
///  root [8] std::string s = (char*)m.gime( "aap" );
///  root [9] s
///  (class TString)"aap"
/// ~~~
/// It is possible to switch between interpreters by calling `TPython::Prompt()`
/// on the Cling side, while returning with `^D` (EOF). State is preserved between
/// successive switches.
///
/// The API part provides (direct) C++ access to the bindings functionality of
/// PyROOT. It allows verifying that you deal with a PyROOT python object in the
/// first place (CPPInstance_Check for CPPInstance and any derived types, as well
/// as CPPInstance_CheckExact for CPPInstance's only); and it allows conversions
/// of `void*` to an CPPInstance and vice versa.

//- data ---------------------------------------------------------------------
ClassImp(TPython);
static PyObject *gMainDict = 0;

//- static public members ----------------------------------------------------
/// Initialization method: setup the python interpreter and load the
/// ROOT module.
Bool_t TPython::Initialize()
{
   static Bool_t isInitialized = kFALSE;
   if (isInitialized)
      return kTRUE;

   // to trgger the initialization of CPyCppyy, and therefore the
   // initialization of Python
   CPyCppyy::Instance_AsVoidPtr(nullptr);

   // force loading of the ROOT module
   const int ret = PyRun_SimpleString(const_cast<char *>("import ROOT"));
   if( ret != 0 )
   {
       std::cerr << "Error: import ROOT failed, check your PYTHONPATH environmental variable." << std::endl;
       return kFALSE;
   }

   if (!gMainDict) {
      // retrieve the main dictionary
      gMainDict = PyModule_GetDict(PyImport_AddModule(const_cast<char *>("__main__")));
      Py_INCREF(gMainDict);
   }

   // declare success ...
   isInitialized = kTRUE;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Import the named python module and create Cling equivalents for its classes
/// and methods.

Bool_t TPython::Import(const char *mod_name)
{
   return CPyCppyy::Import(mod_name);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the give python script as if it were a macro (effectively an
/// execfile in __main__), and create Cling equivalents for any newly available
/// python classes.

void TPython::LoadMacro(const char *name)
{
   CPyCppyy::LoadMacro(name);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute a python stand-alone script, with argv CLI arguments.
///
/// example of use:
///    const char* argv[] = { "1", "2", "3" };
///    TPython::ExecScript( "test.py", sizeof(argv)/sizeof(argv[0]), argv );

void TPython::ExecScript(const char *name, int argc, const char **argv)
{
   // setup
   if (!Initialize())
      return;

   // verify arguments
   if (!name) {
      std::cerr << "Error: no file name specified." << std::endl;
      return;
   }

   std::vector<std::string> args(argc);
   for (int i = 0; i < argc; ++i) {
      args[i] = argv[i];
   }
   CPyCppyy::ExecScript(name, args);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute a python statement (e.g. "import ROOT").

Bool_t TPython::Exec(const char *cmd)
{
   return CPyCppyy::Exec(cmd);
}

////////////////////////////////////////////////////////////////////////////////
/// Evaluate a python expression (e.g. "ROOT.TBrowser()").
///
/// Caution: do not hold on to the return value: either store it in a builtin
/// type (implicit casting will work), or in a pointer to a ROOT object (explicit
/// casting to a void* is required).

const TPyReturn TPython::Eval(const char *expr)
{
   // to implicitly import ROOT
   if(!Initialize())
      return TPyReturn{};

   return TPyReturn{static_cast<PyObject *>(CPyCppyy::Eval(expr))};
}

////////////////////////////////////////////////////////////////////////////////
/// Bind a ROOT object with, at the python side, the name "label".

Bool_t TPython::Bind(TObject *object, const char *label)
{
   // bind object in the main namespace
   TClass *klass = object->IsA();
   if (klass != 0) {
      PyObject *bound = CPyCppyy::Instance_FromVoidPtr((void *)object, klass->GetName());

      if (bound) {
         // to initialize gMainDict
         if(!Initialize())
            return false;
         Bool_t bOk = PyDict_SetItemString(gMainDict, const_cast<char *>(label), bound) == 0;
         Py_DECREF(bound);

         return bOk;
      }
   }

   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Enter an interactive python session (exit with ^D). State is preserved
/// between successive calls.

void TPython::Prompt()
{
   CPyCppyy::Prompt();
}

////////////////////////////////////////////////////////////////////////////////
/// Test whether the type of the given pyobject is of CPPInstance type or any
/// derived type.

Bool_t TPython::CPPInstance_Check(PyObject *pyobject)
{
   return CPyCppyy::Instance_Check(pyobject);
}

////////////////////////////////////////////////////////////////////////////////
/// Test whether the type of the given pyobject is CPPinstance type.

Bool_t TPython::CPPInstance_CheckExact(PyObject *pyobject)
{
   return CPyCppyy::Instance_CheckExact(pyobject);
}

////////////////////////////////////////////////////////////////////////////////
/// Test whether the type of the given pyobject is of CPPOverload type or any
/// derived type.

Bool_t TPython::CPPOverload_Check(PyObject *pyobject)
{
   return CPyCppyy::Overload_Check(pyobject);
}

////////////////////////////////////////////////////////////////////////////////
/// Test whether the type of the given pyobject is CPPOverload type.

Bool_t TPython::CPPOverload_CheckExact(PyObject *pyobject)
{
   return CPyCppyy::Overload_CheckExact(pyobject);
}

////////////////////////////////////////////////////////////////////////////////
/// Extract the object pointer held by the CPPInstance pyobject.

void *TPython::CPPInstance_AsVoidPtr(PyObject *pyobject)
{
   return CPyCppyy::Instance_AsVoidPtr(pyobject);
}

////////////////////////////////////////////////////////////////////////////////
/// Bind the addr to a python object of class defined by classname.

PyObject *TPython::CPPInstance_FromVoidPtr(void *addr, const char *classname, Bool_t python_owns)
{
   return CPyCppyy::Instance_FromVoidPtr(addr, classname, python_owns);
}
