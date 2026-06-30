// Bindings
#include "CPyCppyy.h"
#define CPYCPPYY_INTERNAL 1
#include "CPyCppyy/API.h"
#undef CPYCPPYY_INTERNAL

#include "CPPInstance.h"
#include "CPPOverload.h"
#include "CPPScope.h"
#include "CPyCppyy/DispatchPtr.h"
#include "ProxyWrappers.h"
#include "PyStrings.h"

// Standard
#include <stdio.h>
#include <iostream>
#include <string>

//______________________________________________________________________________
//                  CPyCppyy API: Interpreter and Proxy Access
//                  ==========================================
//
// Access to cppyy Python objects from Cling and C++: allows conversion for
// instances and type checking for scopes, instances, etc.
// Adds a few convenience functions to call Python from Cling and expose Python
// classes to Cling for use in inheritance etc.


//- data ---------------------------------------------------------------------
static PyObject* gMainDict = nullptr;

namespace CPyCppyy {
    extern PyObject* gThisModule;
}


//- private helpers ----------------------------------------------------------
namespace {

static bool Initialize()
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
#if PY_VERSION_HEX < 0x03080000
        Py_Initialize();
#else
        PyConfig config;
        PyConfig_InitPythonConfig(&config);
        PyConfig_SetString(&config, &config.program_name, L"cppyy");
        Py_InitializeFromConfig(&config);
#endif
#if PY_VERSION_HEX >= 0x03020000
#if PY_VERSION_HEX < 0x03090000
        PyEval_InitThreads();
#endif
#endif

    // try again to see if the interpreter is initialized
        if (!Py_IsInitialized()) {
        // give up ...
            std::cerr << "Error: python has not been initialized; returning." << std::endl;
            return false;
        }

   // set the command line arguments on python's sys.argv
#if PY_VERSION_HEX < 0x03000000
        char* argv[] = {const_cast<char*>("cppyy")};
#elif PY_VERSION_HEX < 0x03080000
        wchar_t* argv[] = {const_cast<wchar_t*>(L"cppyy")};
#endif
#if PY_VERSION_HEX < 0x03080000
        PySys_SetArgv(sizeof(argv)/sizeof(argv[0]), argv);
#endif

    // force loading of the cppyy module
        PyRun_SimpleString(const_cast<char*>("import cppyy"));
    }

    if (!gMainDict) {
        CPyCppyy::PythonGILRAII python_gil_raii;
    // retrieve the main dictionary
        gMainDict = PyModule_GetDict(
            PyImport_AddModule(const_cast<char*>("__main__")));
    // The gMainDict is borrowed, i.e. we are not calling Py_INCREF(gMainDict).
    // Like this, we avoid unexpectedly affecting how long __main__ is kept
    // alive. The gMainDict is only used in Exec(), ExecScript(), and Eval(),
    // which should not be called after __main__ is garbage collected anyway.
    }

// declare success ...
    isInitialized = true;
    return true;
}

} // unnamed namespace


//- C++ access to cppyy objects ---------------------------------------------
std::string CPyCppyy::Instance_GetScopedFinalName(PyObject* pyobject)
{
   if (!Instance_Check(pyobject)) {
       PyErr_SetString(PyExc_TypeError, "Instance_GetScopedFinalName : object is not a C++ instance");
       return "";
   }

   Cppyy::TCppType_t pyobjectClass = ((CPPInstance *)pyobject)->ObjectIsA();
   return Cppyy::GetScopedFinalName(pyobjectClass);
}

//-----------------------------------------------------------------------------
void* CPyCppyy::Instance_AsVoidPtr(PyObject* pyobject)
{
// Extract the object pointer held by the CPPInstance pyobject.
    if (!Initialize())
        return nullptr;

    PythonGILRAII python_gil_raii;

// check validity of cast
    if (!CPPInstance_Check(pyobject))
        return nullptr;

// get held object (may be null)
    return ((CPPInstance*)pyobject)->GetObject();
}

//-----------------------------------------------------------------------------
PyObject* CPyCppyy::Instance_FromVoidPtr(
    void* addr, const std::string& classname, bool python_owns)
{
// Bind the addr to a python object of class defined by classname.
    if (!Initialize())
        return nullptr;

    PythonGILRAII python_gil_raii;

// perform cast (the call will check TClass and addr, and set python errors)
    PyObject* pyobject = BindCppObjectNoCast(addr, Cppyy::GetScope(classname), false);

// give ownership, for ref-counting, to the python side, if so requested
    if (python_owns && CPPInstance_Check(pyobject))
        ((CPPInstance*)pyobject)->PythonOwns();

    return pyobject;
}

//-----------------------------------------------------------------------------
PyObject* CPyCppyy::Instance_FromVoidPtr(
    void* addr, Cppyy::TCppScope_t klass_scope, bool python_owns)
{
// Bind the addr to a python object of class defined by classname.
    if (!Initialize())
        return nullptr;

    PythonGILRAII python_gil_raii;

// perform cast (the call will check TClass and addr, and set python errors)
    PyObject* pyobject = BindCppObjectNoCast(addr, klass_scope, false);

// give ownership, for ref-counting, to the python side, if so requested
    if (python_owns && CPPInstance_Check(pyobject))
        ((CPPInstance*)pyobject)->PythonOwns();

    return pyobject;
}
namespace CPyCppyy {
// version with C type arguments only for use with Numba
PyObject* Instance_FromVoidPtr(void* addr, const char* classname, int python_owns) {
    return Instance_FromVoidPtr(addr, std::string(classname), (bool)python_owns);
}
} // namespace CPyCppyy

//-----------------------------------------------------------------------------
bool CPyCppyy::Scope_Check(PyObject* pyobject)
{
// Test if the given object is of a CPPScope derived type.
    if (!Initialize())
        return false;

    PythonGILRAII python_gil_raii;
    return CPPScope_Check(pyobject);
}

//-----------------------------------------------------------------------------
bool CPyCppyy::Scope_CheckExact(PyObject* pyobject)
{
// Test if the given object is of a CPPScope type.
    if (!Initialize())
        return false;

    PythonGILRAII python_gil_raii;
    return CPPScope_CheckExact(pyobject);
}

//-----------------------------------------------------------------------------
bool CPyCppyy::Instance_Check(PyObject* pyobject)
{
// Test if the given pyobject is of CPPInstance derived type.
    if (!Initialize())
        return false;

    PythonGILRAII python_gil_raii;    
// detailed walk through inheritance hierarchy
    return CPPInstance_Check(pyobject);
}

//-----------------------------------------------------------------------------
bool CPyCppyy::Instance_CheckExact(PyObject* pyobject)
{
// Test if the given pyobject is of CPPInstance type.
    if (!Initialize())
        return false;

    PythonGILRAII python_gil_raii;
// direct pointer comparison of type member
    return CPPInstance_CheckExact(pyobject);
}

//-----------------------------------------------------------------------------
void CPyCppyy::Instance_SetPythonOwns(PyObject* pyobject)
{
    if (!Initialize())
        return;

// check validity of cast
    if (!CPPInstance_Check(pyobject))
        return;

    ((CPPInstance *)pyobject)->PythonOwns();
}

//-----------------------------------------------------------------------------
void CPyCppyy::Instance_SetCppOwns(PyObject* pyobject)
{
    if (!Initialize())
        return;

// check validity of cast
    if (!CPPInstance_Check(pyobject))
        return;

    ((CPPInstance *)pyobject)->CppOwns();
}

//-----------------------------------------------------------------------------
bool CPyCppyy::Sequence_Check(PyObject* pyobject)
{
    PythonGILRAII python_gil_raii;
// Extends on PySequence_Check() to determine whether an object can be iterated
// over (technically, all objects can b/c of C++ pointer arithmetic, hence this
// check isn't 100% accurate, but neither is PySequence_Check()).

// Note: simply having the iterator protocol does not constitute a sequence, bc
// PySequence_GetItem() would fail.

// default to PySequence_Check() if called with a non-C++ object
    if (!CPPInstance_Check(pyobject))
        return (bool)PySequence_Check(pyobject);

// all C++ objects should have sq_item defined, but a user-derived class may
// have deleted it, in which case this is not a sequence
    PyTypeObject* t = Py_TYPE(pyobject);
    if (!t->tp_as_sequence || !t->tp_as_sequence->sq_item)
        return false;

// if this is the default getitem, it is only a sequence if it's an array type
    if (t->tp_as_sequence->sq_item == CPPInstance_Type.tp_as_sequence->sq_item) {
        if (((CPPInstance*)pyobject)->fFlags & CPPInstance::kIsArray)
            return true;
        return false;
    }

// TODO: could additionally verify whether __len__ is supported and/or whether
// operator()[] takes an int argument type

    return true;
}

//-----------------------------------------------------------------------------
bool CPyCppyy::Instance_IsLively(PyObject* pyobject)
{
    PythonGILRAII python_gil_raii;
// Test whether the given instance can safely return to C++
    if (!CPPInstance_Check(pyobject))
        return true;    // simply don't know

// the instance fails the lively test if it owns the C++ object while having a
// reference count of 1 (meaning: it could delete the C++ instance any moment)
    if (Py_REFCNT(pyobject) <= 1 && (((CPPInstance*)pyobject)->fFlags & CPPInstance::kIsOwner))
        return false;

    return true;
}

//-----------------------------------------------------------------------------
bool CPyCppyy::Overload_Check(PyObject* pyobject)
{
// Test if the given pyobject is of CPPOverload derived type.
    if (!Initialize())
        return false;

    PythonGILRAII python_gil_raii;
// detailed walk through inheritance hierarchy
    return CPPOverload_Check(pyobject);
}

//-----------------------------------------------------------------------------
bool CPyCppyy::Overload_CheckExact(PyObject* pyobject)
{
// Test if the given pyobject is of CPPOverload type.
    if (!Initialize())
        return false;

    PythonGILRAII python_gil_raii;
// direct pointer comparison of type member
    return CPPOverload_CheckExact(pyobject);
}

//-----------------------------------------------------------------------------
void CPyCppyy::Instance_SetReduceMethod(PyCFunction reduceMethod)
{
    CPPInstance::ReduceMethod() = reduceMethod;
}

//- access to the python interpreter ----------------------------------------
bool CPyCppyy::Import(const std::string& mod_name)
{
// Import the named python module and create Cling equivalents for its classes.
    if (!Initialize())
        return false;

    PythonGILRAII python_gil_raii;

    PyObject* mod = PyImport_ImportModule(mod_name.c_str());
    if (!mod) {
        PyErr_Print();
        return false;
    }

// allow finding to prevent creation of a python proxy for the C++ proxy
    Py_INCREF(mod);
    PyModule_AddObject(gThisModule, mod_name.c_str(), mod);

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
        if (PyClass_Check(value) || PyObject_HasAttr(value, PyStrings::gBases)) {
        // get full class name (including module)
            PyObject* pyClName = PyObject_GetAttr(value, PyStrings::gName);
            if (PyErr_Occurred())
                PyErr_Clear();

        // build full, qualified name
            std::string fullname = mod_name;
            fullname += ".";
            fullname += CPyCppyy_PyText_AsString(pyClName);

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
void CPyCppyy::ExecScript(const std::string& name, const std::vector<std::string>& args)
{
// Execute a python stand-alone script, with argv CLI arguments.
//
// example of use:
//    CPyCppyy::ExecScript("test.py", {"1", "2", "3"});

    if (!Initialize())
        return;

    PythonGILRAII python_gil_raii;

// verify arguments
    if (name.empty()) {
        std::cerr << "Error: no file name specified." << std::endl;
        return;
    }

    FILE* fp = fopen(name.c_str(), "r");
    if (!fp) {
        std::cerr << "Error: could not open file \"" << name << "\"." << std::endl;
        return;
    }

// store a copy of the old cli for restoration
    PyObject* oldargv = PySys_GetObject("argv");   // borrowed
    if (oldargv) {
        PyObject* copy = PyList_GetSlice(oldargv, 0, PyList_Size(oldargv));
        oldargv = copy;  // now owned
    } else {
        PyErr_Clear();
    }

// build new argv
    const int argc = (int)args.size() + 1;
    std::vector<wchar_t*> wargv(argc);
    wargv[0] = Py_DecodeLocale(name.c_str(), nullptr);

    for (int i = 1; i < argc; ++i) {
        wargv[i] = Py_DecodeLocale(args[i - 1].c_str(), nullptr);
    }

// set sys.argv
    PyObject* sysmod = PyImport_ImportModule("sys");   // new reference
    if (sysmod) {
        PyObject* argv_obj = PyList_New(argc);
        for (int i = 0; i < argc; ++i) {
            PyList_SET_ITEM(argv_obj, i, PyUnicode_FromWideChar(wargv[i], -1));
        }
        PyObject_SetAttrString(sysmod, "argv", argv_obj);
        Py_DECREF(argv_obj);
        Py_DECREF(sysmod);
    } else {
        PyErr_Print();
    }

// actual script execution
    PyObject* gbl = PyDict_Copy(gMainDict);
    PyObject* result =       // PyRun_FileEx closes fp (b/c of last argument "1")
        PyRun_FileEx(fp, const_cast<char*>(name.c_str()), Py_file_input, gbl, gbl, 1);

    if (!result)
        PyErr_Print();

    Py_XDECREF(result);
    Py_DECREF(gbl);

// restore original command line
    if (oldargv) {
        PySys_SetObject("argv", oldargv);
        Py_DECREF(oldargv);
    }

// free memory from Py_DecodeLocale
    for (auto ptr : wargv)
        PyMem_RawFree(ptr);
}

//-----------------------------------------------------------------------------
bool CPyCppyy::Exec(const std::string& cmd)
{
// Execute a python statement (e.g. "import noddy").
    if (!Initialize())
        return false;

    PythonGILRAII python_gil_raii;
// execute the command
    PyObject* result =
        PyRun_String(const_cast<char*>(cmd.c_str()), Py_file_input, gMainDict, gMainDict);

// test for error
    if (result) {
        Py_DECREF(result);
        return true;
    }

    PyErr_Print();
    return false;
}

//-----------------------------------------------------------------------------
void CPyCppyy::Prompt() {
// Enter an interactive python session (exit with ^D). State is preserved
// between successive calls.
    if (!Initialize())
        return;

    PythonGILRAII python_gil_raii;
// enter i/o interactive mode
    PyRun_InteractiveLoop(stdin, const_cast<char*>("\0"));
}
