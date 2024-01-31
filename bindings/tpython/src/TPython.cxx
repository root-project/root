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
#include "TPyClassGenerator.h"

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

namespace {

class CachedPyString {

public:
   CachedPyString(const char *name) : fObj{PyUnicode_FromString(name)} {}

   CachedPyString(CachedPyString const&) = delete;
   CachedPyString(CachedPyString &&) = delete;
   CachedPyString& operator=(CachedPyString const&) = delete;
   CachedPyString& operator=(CachedPyString &&) = delete;

   ~CachedPyString() { Py_DECREF(fObj); }

   PyObject *obj() { return fObj; }

private:
   PyObject *fObj = nullptr;
};

namespace PyStrings {
PyObject *basesStr()
{
   static CachedPyString wrapper{"__bases__"};
   return wrapper.obj();
}
PyObject *cppNameStr()
{
   static CachedPyString wrapper{"__cpp_name__"};
   return wrapper.obj();
}
PyObject *moduleStr()
{
   static CachedPyString wrapper{"__module__"};
   return wrapper.obj();
}
PyObject *nameStr()
{
   static CachedPyString wrapper{"__name__"};
   return wrapper.obj();
}
} // namespace PyStrings

} // namespace

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
   if (!CPyCppyy::Import(mod_name)) {
      return false;
   }

   // force creation of the module as a namespace
   TClass::GetClass(mod_name, kTRUE);

   PyObject *modNameObj = PyUnicode_FromString(mod_name);
   PyObject *mod = PyImport_GetModule(modNameObj);
   PyObject *dct = PyModule_GetDict(mod);

   // create Cling classes for all new python classes
   PyObject *values = PyDict_Values(dct);
   for (int i = 0; i < PyList_GET_SIZE(values); ++i) {
      PyObject *value = PyList_GET_ITEM(values, i);
      Py_INCREF(value);

      // collect classes
      if (PyType_Check(value) || PyObject_HasAttr(value, PyStrings::basesStr())) {
         // get full class name (including module)
         PyObject *pyClName = PyObject_GetAttr(value, PyStrings::cppNameStr());
         if (!pyClName) {
            pyClName = PyObject_GetAttr(value, PyStrings::nameStr());
         }

         if (PyErr_Occurred())
            PyErr_Clear();

         // build full, qualified name
         std::string fullname = mod_name;
         fullname += ".";
         fullname += PyUnicode_AsUTF8(pyClName);

         // force class creation (this will eventually call TPyClassGenerator)
         TClass::GetClass(fullname.c_str(), kTRUE);

         Py_XDECREF(pyClName);
      }

      Py_DECREF(value);
   }

   Py_DECREF(values);
   Py_DECREF(mod);
   Py_DECREF(modNameObj);

   if (PyErr_Occurred())
      return kFALSE;
   return kTRUE;
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

   // obtain a reference to look for new classes later
   PyObject *old = PyDict_Values(gMainDict);

// actual execution
#if PY_VERSION_HEX < 0x03000000
   Exec((std::string("execfile(\"") + name + "\")").c_str());
#else
   Exec((std::string("__pyroot_f = open(\"") + name + "\"); "
                                                      "exec(__pyroot_f.read()); "
                                                      "__pyroot_f.close(); del __pyroot_f")
           .c_str());
#endif

   // obtain new __main__ contents
   PyObject *current = PyDict_Values(gMainDict);

   // create Cling classes for all new python classes
   for (int i = 0; i < PyList_GET_SIZE(current); ++i) {
      PyObject *value = PyList_GET_ITEM(current, i);
      Py_INCREF(value);

      if (!PySequence_Contains(old, value)) {
         // collect classes
         if (PyType_Check(value) || PyObject_HasAttr(value, PyStrings::basesStr())) {
            // get full class name (including module)
            PyObject *pyModName = PyObject_GetAttr(value, PyStrings::moduleStr());
            PyObject *pyClName = PyObject_GetAttr(value, PyStrings::nameStr());

            if (PyErr_Occurred())
               PyErr_Clear();

            // need to check for both exact and derived (differences exist between older and newer
            // versions of python ... bug?)
            if ((pyModName && pyClName) &&
                ((PyUnicode_CheckExact(pyModName) && PyUnicode_CheckExact(pyClName)) ||
                 (PyUnicode_Check(pyModName) && PyUnicode_Check(pyClName)))) {
               // build full, qualified name
               std::string fullname = PyUnicode_AsUTF8(pyModName);
               fullname += '.';
               fullname += PyUnicode_AsUTF8(pyClName);

               // force class creation (this will eventually call TPyClassGenerator)
               TClass::GetClass(fullname.c_str(), kTRUE);
            }

            Py_XDECREF(pyClName);
            Py_XDECREF(pyModName);
         }
      }

      Py_DECREF(value);
   }

   Py_DECREF(current);
   Py_DECREF(old);
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

   FILE *fp = fopen(name, "r");
   if (!fp) {
      std::cerr << "Error: could not open file \"" << name << "\"." << std::endl;
      return;
   }

   // store a copy of the old cli for restoration
   PyObject *oldargv = PySys_GetObject(const_cast<char *>("argv")); // borrowed
   if (!oldargv)                                                    // e.g. apache
      PyErr_Clear();
   else {
      PyObject *l = PyList_New(PyList_GET_SIZE(oldargv));
      for (int i = 0; i < PyList_GET_SIZE(oldargv); ++i) {
         PyObject *item = PyList_GET_ITEM(oldargv, i);
         Py_INCREF(item);
         PyList_SET_ITEM(l, i, item); // steals ref
      }
      oldargv = l;
   }

   // create and set (add progam name) the new command line
   argc += 1;
   // This is a common block for Python 3. We prefer using objects to automatize memory management and not introduce
   // even more preprocessor branching for deletion at the end of the method.
   // FUTURE IMPROVEMENT ONCE OLD PYTHON VERSIONS ARE NOT SUPPORTED BY ROOT:
   // Right now we use C++ objects to automatize memory management. One could use RAAI and the Python memory allocation
   // API (PEP 445) once some old Python version is deprecated in ROOT. That new feature is available since version 3.4
   // and the preprocessor branching to also support that would be so complicated to make the code unreadable.
   std::vector<std::wstring> argv2;
   argv2.reserve(argc);
   argv2.emplace_back(name, &name[strlen(name)]);

   for (int i = 1; i < argc; ++i) {
      auto iarg = argv[i - 1];
      argv2.emplace_back(iarg, &iarg[strlen(iarg)]);
   }

#if PY_VERSION_HEX < 0x03080000
   // Before version 3.8, the code is one simple line
   wchar_t *argv2_arr[argc];
   for (int i = 0; i < argc; ++i) {
      argv2_arr[i] = const_cast<wchar_t *>(argv2[i].c_str());
   }
   PySys_SetArgv(argc, argv2_arr);

#else
   // Here we comply to "PEP 587 – Python Initialization Configuration" to avoid deprecation warnings at compile time.
   class PyConfigHelperRAAI {
   public:
      PyConfigHelperRAAI(const std::vector<std::wstring> &argv2)
      {
         PyConfig_InitPythonConfig(&fConfig);
         fConfig.parse_argv = 1;
         UpdateArgv(argv2);
         InitFromConfig();
      }
      ~PyConfigHelperRAAI() { PyConfig_Clear(&fConfig); }

   private:
      void InitFromConfig() { Py_InitializeFromConfig(&fConfig); };
      void UpdateArgv(const std::vector<std::wstring> &argv2)
      {
         auto WideStringListAppendHelper = [](PyWideStringList *wslist, const wchar_t *wcstr) {
            PyStatus append_status = PyWideStringList_Append(wslist, wcstr);
            if (PyStatus_IsError(append_status)) {
               std::wcerr << "Error: could not append element " << wcstr << " to arglist - " << append_status.err_msg
                          << std::endl;
            }
         };
         WideStringListAppendHelper(&fConfig.argv, Py_GetProgramName());
         for (const auto &iarg : argv2) {
            WideStringListAppendHelper(&fConfig.argv, iarg.c_str());
         }
      }
      PyConfig fConfig;
   };

   PyConfigHelperRAAI pych(argv2);

#endif // of the else branch of PY_VERSION_HEX < 0x03080000

   // actual script execution
   PyObject *gbl = PyDict_Copy(gMainDict);
   PyObject *result = // PyRun_FileEx closes fp (b/c of last argument "1")
      PyRun_FileEx(fp, const_cast<char *>(name), Py_file_input, gbl, gbl, 1);
   if (!result) {
      std::cerr << "An error occurred executing file " << name << std::endl;
      PyErr_Print();
   }

   Py_XDECREF(result);
   Py_DECREF(gbl);

   // restore original command line
   if (oldargv) {
      PySys_SetObject(const_cast<char *>("argv"), oldargv);
      Py_DECREF(oldargv);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Execute a python statement (e.g. "import ROOT").

Bool_t TPython::Exec(const char *cmd)
{
   // setup
   if (!Initialize())
      return kFALSE;

   // execute the command
   PyObject *result = PyRun_String(const_cast<char *>(cmd), Py_file_input, gMainDict, gMainDict);

   // test for error
   if (result) {
      Py_DECREF(result);
      return kTRUE;
   }

   PyErr_Print();
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Evaluate a python expression (e.g. "ROOT.TBrowser()").
///
/// Caution: do not hold on to the return value: either store it in a builtin
/// type (implicit casting will work), or in a pointer to a ROOT object (explicit
/// casting to a void* is required).

const TPyReturn TPython::Eval(const char *expr)
{
   // setup
   if (!Initialize())
      return TPyReturn();

   // evaluate the expression
   PyObject *result = PyRun_String(const_cast<char *>(expr), Py_eval_input, gMainDict, gMainDict);

   // report errors as appropriate; return void
   if (!result) {
      PyErr_Print();
      return TPyReturn();
   }

   // results that require no conversion
   if (result == Py_None || CPyCppyy::Instance_Check(result) || PyBytes_Check(result) || PyFloat_Check(result) ||
       PyLong_Check(result))
      return TPyReturn(result);

   // explicit conversion for python type required
   PyObject *pyclass = PyObject_GetAttrString(result, const_cast<char*>("__class__"));
   if (pyclass != 0) {
      // retrieve class name and the module in which it resides
      PyObject *name = PyObject_GetAttr(pyclass, PyStrings::nameStr());
      PyObject *module = PyObject_GetAttr(pyclass, PyStrings::moduleStr());

      // concat name
      std::string qname = std::string(PyUnicode_AsUTF8(module)) + '.' + PyUnicode_AsUTF8(name);
      Py_DECREF(module);
      Py_DECREF(name);
      Py_DECREF(pyclass);

      // locate ROOT style class with this name
      TClass *klass = TClass::GetClass(qname.c_str());

      // construct general ROOT python object that pretends to be of class 'klass'
      if (klass != 0)
         return TPyReturn(result);
   } else
      PyErr_Clear();

   // no conversion, return null pointer object
   Py_DECREF(result);
   return TPyReturn();
}

////////////////////////////////////////////////////////////////////////////////
/// Bind a ROOT object with, at the python side, the name "label".

Bool_t TPython::Bind(TObject *object, const char *label)
{
   // check given address and setup
   if (!(object && Initialize()))
      return kFALSE;

   // bind object in the main namespace
   TClass *klass = object->IsA();
   if (klass != 0) {
      PyObject *bound = CPyCppyy::Instance_FromVoidPtr((void *)object, klass->GetName());

      if (bound) {
         Bool_t bOk = PyDict_SetItemString(gMainDict, const_cast<char *>(label), bound) == 0;
         Py_DECREF(bound);

         return bOk;
      }
   }

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Enter an interactive python session (exit with ^D). State is preserved
/// between successive calls.

void TPython::Prompt()
{
   // setup
   if (!Initialize()) {
      return;
   }

   // enter i/o interactive mode
   PyRun_InteractiveLoop(stdin, const_cast<char *>("\0"));
}

////////////////////////////////////////////////////////////////////////////////
/// Test whether the type of the given pyobject is of CPPInstance type or any
/// derived type.

Bool_t TPython::CPPInstance_Check(PyObject *pyobject)
{
   // setup
   if (!Initialize())
      return kFALSE;

   // detailed walk through inheritance hierarchy
   return CPyCppyy::Instance_Check(pyobject);
}

////////////////////////////////////////////////////////////////////////////////
/// Test whether the type of the given pyobject is CPPinstance type.

Bool_t TPython::CPPInstance_CheckExact(PyObject *pyobject)
{
   // setup
   if (!Initialize())
      return kFALSE;

   // direct pointer comparison of type member
   return CPyCppyy::Instance_CheckExact(pyobject);
}

////////////////////////////////////////////////////////////////////////////////
/// Test whether the type of the given pyobject is of CPPOverload type or any
/// derived type.

Bool_t TPython::CPPOverload_Check(PyObject *pyobject)
{
   // setup
   if (!Initialize())
      return kFALSE;

   // detailed walk through inheritance hierarchy
   return CPyCppyy::Overload_Check(pyobject);
}

////////////////////////////////////////////////////////////////////////////////
/// Test whether the type of the given pyobject is CPPOverload type.

Bool_t TPython::CPPOverload_CheckExact(PyObject *pyobject)
{
   // setup
   if (!Initialize())
      return kFALSE;

   // direct pointer comparison of type member
   return CPyCppyy::Overload_CheckExact(pyobject);
}

////////////////////////////////////////////////////////////////////////////////
/// Extract the object pointer held by the CPPInstance pyobject.

void *TPython::CPPInstance_AsVoidPtr(PyObject *pyobject)
{
   // setup
   if (!Initialize())
      return 0;

   // get held object (may be null)
   return CPyCppyy::Instance_AsVoidPtr(pyobject);
}

////////////////////////////////////////////////////////////////////////////////
/// Bind the addr to a python object of class defined by classname.

PyObject *TPython::CPPInstance_FromVoidPtr(void *addr, const char *classname, Bool_t python_owns)
{
   // setup
   if (!Initialize())
      return 0;

   // perform cast (the call will check TClass and addr, and set python errors)
   // give ownership, for ref-counting, to the python side, if so requested
   return CPyCppyy::Instance_FromVoidPtr(addr, classname, python_owns);
}
