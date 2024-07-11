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
#include "../../pyroot/cppyy/CPyCppyy/src/TPyClassGenerator.h"

#include "TPython.h"

// ROOT
#include "TClass.h"
#include "TClassRef.h"
#include "TObject.h"
#include "TROOT.h"

#include <Riostream.h>

// Standard
#include <sstream>
#include <stdio.h>
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
///  root [0] TPython::Exec( "print('Hello World!')" );
///  Hello World!
///
///  // Create a TNamed on the python side, and transfer it back and forth.
///  root [1] std::any res1;
///  root [2] TPython::Exec("_anyresult = ROOT.std.make_any['TNamed']('hello', '')", &res1);
///  root [3] TPython::Bind(&std::any_cast<TNamed&>(res1), "n");
///  root [4] std::any res2;
///  root [5] TPython::Exec("_anyresult = ROOT.std.make_any['TNamed*', 'TNamed*'](n)", &res2);
///  root [6] (&std::any_cast<TNamed&>(res1) == std::any_cast<TNamed*>(res2))
///  (bool) true
///
///  // Variables can cross-over by using an `std::any` with a specific name.
///  root [6] TPython::Exec("_anyresult = ROOT.std.make_any['Int_t'](1 + 1)", &res1);
///  root [7] std::any_cast<int>(res1)
///  (int) 2
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

   if (!Py_IsInitialized()) {
// this happens if Cling comes in first
#if PY_VERSION_HEX < 0x03020000
      PyEval_InitThreads();
#endif

// set the command line arguments on python's sys.argv
#if PY_VERSION_HEX < 0x03000000
      char *argv[] = {const_cast<char *>("root")};
#else
      wchar_t *argv[] = {const_cast<wchar_t *>(L"root")};
#endif
      int argc = sizeof(argv) / sizeof(argv[0]);
#if PY_VERSION_HEX < 0x030b0000
      Py_Initialize();
#else
      PyStatus status;
      PyConfig config;

      PyConfig_InitPythonConfig(&config);

      status = PyConfig_SetArgv(&config, argc, argv);
      if (PyStatus_Exception(status)) {
         PyConfig_Clear(&config);
         std::cerr << "Error when setting command line arguments." << std::endl;
         return kFALSE;
      }

      status = Py_InitializeFromConfig(&config);
      if (PyStatus_Exception(status)) {
         PyConfig_Clear(&config);
         std::cerr << "Error when initializing Python." << std::endl;
         return kFALSE;
      }
      PyConfig_Clear(&config);
#endif
#if PY_VERSION_HEX >= 0x03020000
#if PY_VERSION_HEX < 0x03090000
      PyEval_InitThreads();
#endif
#endif

      // try again to see if the interpreter is initialized
      if (!Py_IsInitialized()) {
         // give up ...
         std::cerr << "Error: python has not been intialized; returning." << std::endl;
         return kFALSE;
      }

#if PY_VERSION_HEX < 0x030b0000
      PySys_SetArgv(argc, argv);
#endif

      // force loading of the ROOT module
      const int ret = PyRun_SimpleString(const_cast<char *>("import ROOT"));
      if( ret != 0 )
      {
          std::cerr << "Error: import ROOT failed, check your PYTHONPATH environmental variable." << std::endl;
          return kFALSE;
      }
   }

   if (!gMainDict) {
      // retrieve the main dictionary
      gMainDict = PyModule_GetDict(PyImport_AddModule(const_cast<char *>("__main__")));
      Py_INCREF(gMainDict);
   }

   // python side class construction, managed by ROOT
   gROOT->AddClassGenerator(new TPyClassGenerator);

   // declare success ...
   isInitialized = kTRUE;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Import the named python module and create Cling equivalents for its classes
/// and methods.

Bool_t TPython::Import(const char *mod_name)
{
   // setup
   if (!Initialize())
      return false;

   return CPyCppyy::Import(mod_name);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the give python script as if it were a macro (effectively an
/// execfile in __main__), and create Cling equivalents for any newly available
/// python classes.

void TPython::LoadMacro(const char *name)
{
   // setup
   if (!Initialize())
      return;

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
/// Executes a Python command within the current Python environment.
///
/// This function initializes the Python environment if it is not already
/// initialized. It then executes the specified Python command string using the
/// Python C API.
///
/// In the Python command, you can change the value of a special TPyResult
/// object returned by TPyBuffer(). If the optional result parameter is
/// non-zero, the result parameter will be swapped with a std::any variable on
/// the Python side. You need to define this variable yourself, and it needs to
/// be of type std::any and its name needs to be `"_anyresult"` by default.
/// Like this, you can pass information from Python back to C++.
///
/// \param cmd The Python command to be executed as a string.
/// \param result Optional pointer to a std::any object that can be used to
///               transfer results from Python to C++.
/// \param resultName Name of the Python variable that is swapped over to the std::any result.
///                   The default value is `"_anyresult"`.
/// \return bool Returns `true` if the command was successfully executed,
///              otherwise returns `false`.

Bool_t TPython::Exec(const char *cmd, std::any *result, std::string const &resultName)
{
   bool ret = CPyCppyy::Exec(cmd);
   // Swap the std::any with the one in the C++ world if required
   if (ret && result) {
      std::stringstream command;
      command << "; ROOT.Internal.SwapWithObjAtAddr['std::any'](" << resultName << ", " << reinterpret_cast<std::intptr_t>(result) << ")";
      PyRun_String(const_cast<char *>(command.str().c_str()), Py_file_input, gMainDict, gMainDict);
   }
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Evaluate a python expression (e.g. "ROOT.TBrowser()").
///
/// Caution: do not hold on to the return value: either store it in a builtin
/// type (implicit casting will work), or in a pointer to a ROOT object (explicit
/// casting to a void* is required).
///
/// \deprecated Use TPython::Exec() with an std::any output parameter instead.

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
