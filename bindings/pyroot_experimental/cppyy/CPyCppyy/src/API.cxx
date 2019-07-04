// Bindings
#include "CPyCppyy.h"
#include "CPyCppyy/API.h"

#include "CPPInstance.h"
#include "CPPOverload.h"
#include "ProxyWrappers.h"
#include "PyStrings.h"

// Standard
#include <stdio.h>
#include <iostream>
#include <string>

//______________________________________________________________________________
//                          Python interpreter access
//                          =========================
//
// The TPython class allows for access to python objects from Cling. The current
// functionality is only basic: cppyy objects and builtin types can freely cross
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
//     def __init__(self):
//        print 'in MyPyClass.__init__'
//
//     def gime(self, what):
//        return what
//
//  $ root -l
//  // Execute a string of python code.
//  root [0] TPython::Exec("print \'Hello World!\'");
//  Hello World!
//
//  // Create a TBrowser on the python side, and transfer it back and forth.
//  // Note the required explicit (void*) cast!
//  root [1] TBrowser* b = (void*)TPython::Eval("ROOT.TBrowser()");
//  root [2] TPython::Bind(b, "b");
//  root [3] b == (void*) TPython::Eval("b")
//  (int)1
//
//  // Builtin variables can cross-over by using implicit casts.
//  root [4] int i = TPython::Eval("1 + 1");
//  root [5] i
//  (int)2
//
//  // Load a python module with a class definition, and use it. Casts are
//  // necessary as the type information cannot be otherwise derived.
//  root [6] TPython::LoadMacro("MyPyClass.py");
//  creating class MyPyClass ...
//  root [7] MyPyClass m;
//  in MyPyClass.__init__
//  root [8] std::string s = (char*)m.gime("aap");
//  root [9] s
//  (class TString)"aap"
//
// It is possible to switch between interpreters by calling "TPython::Prompt()"
// on the Cling side, while returning with ^D (EOF). State is preserved between
// successive switches.
//
// The API part provides (direct) C++ access to the bindings functionality of
// CPyCppyy. It allows verifying that you deal with a CPyCppyy pyobject in the
// first place (CPPInstance_Check for CPPInstance and any derived types, as well
// as CPPInstance_CheckExact for CPPInstance's only); and it allows conversions
// of void* to an CPPInstance and vice versa.


//- data ---------------------------------------------------------------------
static PyObject* gMainDict = nullptr;

namespace CPyCppyy {
    extern PyObject* gThisModule;
}


//- static public members ----------------------------------------------------
bool TPython::Initialize()
{
// Private initialization method: setup the python interpreter and load the
// cppyy module.

    static bool isInitialized = false;
    if (isInitialized)
        return true;

    if (!Py_IsInitialized()) {
    // this happens if Cling comes in first
#if PY_VERSION_HEX < 0x03020000
        PyEval_InitThreads();
#endif
        Py_Initialize();
#if PY_VERSION_HEX >= 0x03020000
        PyEval_InitThreads();
#endif

    // try again to see if the interpreter is initialized
        if (!Py_IsInitialized()) {
        // give up ...
            std::cerr << "Error: python has not been intialized; returning." << std::endl;
            return false;
        }

   // set the command line arguments on python's sys.argv
#if PY_VERSION_HEX < 0x03000000
        char* argv[] = {const_cast<char*>("cppyy")};
#else
        wchar_t* argv[] = {const_cast<wchar_t*>(L"cppyy")};
#endif
        PySys_SetArgv(sizeof(argv)/sizeof(argv[0]), argv);

    // force loading of the cppyy module
        PyRun_SimpleString(const_cast<char*>("import cppyy"));
    }

    if (!gMainDict) {
    // retrieve the main dictionary
        gMainDict = PyModule_GetDict(
            PyImport_AddModule(const_cast<char*>("__main__")));
        Py_INCREF(gMainDict);
    }

// declare success ...
    isInitialized = true;
    return true;
}

//-----------------------------------------------------------------------------
bool TPython::Import(const char* mod_name)
{
// Import the named python module and create Cling equivalents for its classes
// and methods.
    if (!Initialize())
        return false;

    PyObject* mod = PyImport_ImportModule(mod_name);
    if (!mod) {
        PyErr_Print();
        return false;
    }

// allow finding to prevent creation of a python proxy for the C++ proxy
    Py_INCREF(mod);
    PyModule_AddObject(CPyCppyy::gThisModule, mod_name, mod);

// force creation of the module as a namespace
// TODO: the following is broken (and should live in Cppyy.cxx)
//   TClass::GetClass(mod_name, true);

    PyObject* dct = PyModule_GetDict(mod);

// create Cling classes for all new python classes
    PyObject* values = PyDict_Values(dct);
    for (int i = 0; i < PyList_GET_SIZE(values); ++i) {
        PyObject* value = PyList_GET_ITEM(values, i);
        Py_INCREF(value);

    // collect classes
        if (PyClass_Check(value) || PyObject_HasAttr(value, CPyCppyy::PyStrings::gBases)) {
        // get full class name (including module)
            PyObject* pyClName = PyObject_GetAttr(value, CPyCppyy::PyStrings::gName);
            if (PyErr_Occurred())
                PyErr_Clear();

        // build full, qualified name
            std::string fullname = mod_name;
            fullname += ".";
            fullname += CPyCppyy_PyUnicode_AsString(pyClName);

      // force class creation (this will eventually call TPyClassGenerator)
      // TODO: the following is broken (and should live in Cppyy.cxx) to
      //         TClass::GetClass(fullname.c_str(), true);

            Py_XDECREF(pyClName);
        }

        Py_DECREF(value);
    }

    Py_DECREF(values);

// TODO: mod "leaks" here
    if (PyErr_Occurred())
        return false;
    return true;
}

//-----------------------------------------------------------------------------
void TPython::LoadMacro(const char* name)
{
// Execute the give python script as if it were a macro (effectively an
// execfile in __main__), and create Cling equivalents for any newly available
// python classes.
    if (!Initialize())
        return;

// obtain a reference to look for new classes later
    PyObject* old = PyDict_Values(gMainDict);

// actual execution
#if PY_VERSION_HEX < 0x03000000
    Exec((std::string("execfile(\"") + name + "\")").c_str());
#else
    Exec((std::string("__cpycppyy_f = open(\"") + name + "\"); "
                      "exec(__cpycppyy_f.read()); "
                      "__cpycppyy_f.close(); del __cpycppyy_f").c_str());
#endif

// obtain new __main__ contents
    PyObject* current = PyDict_Values(gMainDict);

// create Cling classes for all new python classes
    for (int i = 0; i < PyList_GET_SIZE(current); ++i) {
        PyObject* value = PyList_GET_ITEM(current, i);
        Py_INCREF(value);

        if (!PySequence_Contains(old, value)) {
        // collect classes
            if (PyClass_Check(value) || PyObject_HasAttr(value, CPyCppyy::PyStrings::gBases)) {
            // get full class name (including module)
                PyObject* pyModName = PyObject_GetAttr(value, CPyCppyy::PyStrings::gModule);
                PyObject* pyClName  = PyObject_GetAttr(value, CPyCppyy::PyStrings::gName);

                if (PyErr_Occurred())
                    PyErr_Clear();

            // need to check for both exact and derived (differences exist between older and newer
            // versions of python ... bug?)
                if ((pyModName && pyClName) && \
                    ((CPyCppyy_PyUnicode_CheckExact(pyModName) && \
                      CPyCppyy_PyUnicode_CheckExact(pyClName)) || \
                     (CPyCppyy_PyUnicode_Check(pyModName) && \
                      CPyCppyy_PyUnicode_Check(pyClName)) \
                   )) {
            // build full, qualified name
               std::string fullname = CPyCppyy_PyUnicode_AsString(pyModName);
               fullname += '.';
               fullname += CPyCppyy_PyUnicode_AsString(pyClName);

           // force class creation (this will eventually call TPyClassGenerator)
           // the following is broken (and should live in Cppyy.cxx)
           // TClass::GetClass(fullname.c_str(), true);
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

//-----------------------------------------------------------------------------
void TPython::ExecScript(const char* name, int argc, const char**
#if PY_VERSION_HEX < 0x03000000
       argv
#endif
   )
{
// Execute a python stand-alone script, with argv CLI arguments.
//
// example of use:
//    const char* argv[] = {"1", "2", "3"};
//    TPython::ExecScript("test.py", sizeof(argv)/sizeof(argv[0]), argv);
    if (!Initialize())
        return;

// verify arguments
    if (!name) {
        std::cerr << "Error: no file name specified." << std::endl;
        return;
    }

    FILE* fp = fopen(name, "r");
    if (!fp) {
        std::cerr << "Error: could not open file \"" << name << "\"." << std::endl;
        return;
    }

// store a copy of the old cli for restoration
    PyObject* oldargv = PySys_GetObject(const_cast<char*>("argv"));   // borrowed
    if (!oldargv)                                 // e.g. apache
        PyErr_Clear();
    else {
        PyObject* l = PyList_New(PyList_GET_SIZE(oldargv));
        for (int i = 0; i < PyList_GET_SIZE(oldargv); ++i) {
            PyObject* item = PyList_GET_ITEM(oldargv, i);
            Py_INCREF(item);
            PyList_SET_ITEM(l, i, item);          // steals ref
        }
        oldargv = l;
    }

// create and set (add progam name) the new command line
    argc += 1;
#if PY_VERSION_HEX < 0x03000000
    const char** argv2 = new const char*[argc];
    for (int i = 1; i < argc; ++i) argv2[i] = argv[i-1];
    argv2[0] = Py_GetProgramName();
    PySys_SetArgv(argc, const_cast<char**>(argv2));
    delete [] argv2;
#else
// TODO: fix this to work like above ...
#endif

// actual script execution
    PyObject* gbl = PyDict_Copy(gMainDict);
    PyObject* result =   // PyRun_FileEx closes fp (b/c of last argument "1")
        PyRun_FileEx(fp, const_cast<char*>(name), Py_file_input, gbl, gbl, 1);
    if (!result)
        PyErr_Print();
    Py_XDECREF(result);
    Py_DECREF(gbl);

// restore original command line
    if (oldargv) {
        PySys_SetObject(const_cast<char*>("argv"), oldargv);
        Py_DECREF(oldargv);
    }
}

//-----------------------------------------------------------------------------
bool TPython::Exec(const char* cmd)
{
// Execute a python statement (e.g. "import noddy").
    if (!Initialize())
        return false;

// execute the command
    PyObject* result =
        PyRun_String(const_cast<char*>(cmd), Py_file_input, gMainDict, gMainDict);

// test for error
    if (result) {
        Py_DECREF(result);
        return true;
    }

    PyErr_Print();
    return false;
}

//-----------------------------------------------------------------------------
const TPyReturn TPython::Eval(const char* expr)
{
// Evaluate a python expression.
//
// Caution: do not hold on to the return value: either store it in a builtin
// type (implicit casting will work), or in a pointer to a cppyy object (explicit
// casting to a void* is required).
    if (!Initialize())
        return TPyReturn();

// evaluate the expression
    PyObject* result =
        PyRun_String(const_cast<char*>(expr), Py_eval_input, gMainDict, gMainDict);

// report errors as appropriate; return void
    if (!result) {
        PyErr_Print();
        return TPyReturn();
    }

// results that require no convserion
    if (result == Py_None || CPyCppyy::CPPInstance_Check(result) ||
            PyBytes_Check(result) ||
            PyFloat_Check(result) || PyLong_Check(result) || PyInt_Check(result))
        return TPyReturn(result);

// explicit conversion for python type required
    PyObject* pyclass = (PyObject*)Py_TYPE(result);

// retrieve class name and the module in which it resides
    PyObject* name = PyObject_GetAttr(pyclass, CPyCppyy::PyStrings::gName);
    PyObject* module = PyObject_GetAttr(pyclass, CPyCppyy::PyStrings::gModule);

 // concat name
    std::string qname =
        std::string(CPyCppyy_PyUnicode_AsString(module)) + \
                    '.' + CPyCppyy_PyUnicode_AsString(name);
    Py_DECREF(module);
    Py_DECREF(name);

// locate cppyy style class with this name
    // TODO: use Cppyy.cxx ...
    //TClass* klass = TClass::GetClass(qname.c_str());
    void* klass = nullptr;

// construct general cppyy python object that pretends to be of class 'klass'
    if (klass)
        return TPyReturn(result);

// no conversion, return null pointer object
    Py_DECREF(result);
    return TPyReturn();
}

//-----------------------------------------------------------------------------
void TPython::Prompt() {
// Enter an interactive python session (exit with ^D). State is preserved
// between successive calls.
    if (!Initialize()) {
        return;
    }

// enter i/o interactive mode
    PyRun_InteractiveLoop(stdin, const_cast<char*>("\0"));
}

//-----------------------------------------------------------------------------
bool TPython::CPPInstance_Check(PyObject* pyobject)
{
// Test whether the type of the given pyobject is of CPPInstance type or any
// derived type.
    if (!Initialize())
        return false;

// detailed walk through inheritance hierarchy
    return CPyCppyy::CPPInstance_Check(pyobject);
}

//-----------------------------------------------------------------------------
bool TPython::CPPInstance_CheckExact(PyObject* pyobject)
{
// Test whether the type of the given pyobject is CPPInstance type.
    if (!Initialize())
        return false;

// direct pointer comparison of type member
    return CPyCppyy::CPPInstance_CheckExact(pyobject);
}

//-----------------------------------------------------------------------------
bool TPython::CPPOverload_Check(PyObject* pyobject)
{
// Test whether the type of the given pyobject is of CPPOverload type or any
// derived type.
    if (!Initialize())
        return false;

// detailed walk through inheritance hierarchy
    return CPyCppyy::CPPOverload_Check(pyobject);
}

//-----------------------------------------------------------------------------
bool TPython::CPPOverload_CheckExact(PyObject* pyobject)
{
// Test whether the type of the given pyobject is CPPOverload type.
    if (!Initialize())
        return false;

// direct pointer comparison of type member
    return CPyCppyy::CPPOverload_CheckExact(pyobject);
}

//-----------------------------------------------------------------------------
void* TPython::CPPInstance_AsVoidPtr(PyObject* pyobject)
{
// Extract the object pointer held by the CPPInstance pyobject.
    if (!Initialize())
        return nullptr;

// check validity of cast
    if (!CPyCppyy::CPPInstance_Check(pyobject))
        return nullptr;

// get held object (may be null)
    return ((CPyCppyy::CPPInstance*)pyobject)->GetObject();
}

//-----------------------------------------------------------------------------
PyObject* TPython::CPPInstance_FromVoidPtr(
    void* addr, const char* classname, bool python_owns)
{
// Bind the addr to a python object of class defined by classname.
    if (!Initialize())
        return nullptr;

// perform cast (the call will check TClass and addr, and set python errors)
    PyObject* pyobject = CPyCppyy::BindCppObjectNoCast(addr, Cppyy::GetScope(classname), false);

// give ownership, for ref-counting, to the python side, if so requested
    if (python_owns && CPyCppyy::CPPInstance_Check(pyobject))
        ((CPyCppyy::CPPInstance*)pyobject)->PythonOwns();

    return pyobject;
}
