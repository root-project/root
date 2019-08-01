#ifndef CPYCPPYY_TPYTHON
#define CPYCPPYY_TPYTHON

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TPython                                                                  //
//                                                                          //
// Access to the python interpreter and API onto CPyCppyy.                  //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

// Bindings
#include "TPyReturn.h"
#include "CPyCppyy/CommonDefs.h"


class CPYCPPYY_CLASS_EXPORT TPython {

private:
    static bool Initialize();

public:
// import a python module, making its classes available
    static bool Import(const char* name);

// load a python script as if it were a macro
    static void LoadMacro(const char* name);

// execute a python stand-alone script, with argv CLI arguments
    static void ExecScript(const char* name, int argc = 0, const char** argv = 0);

// execute a python statement (e.g. "import sys")
    static bool Exec(const char* cmd);

// evaluate a python expression (e.g. "1+1")
    static const TPyReturn Eval(const char* expr);

// enter an interactive python session (exit with ^D)
    static void Prompt();

// type verifiers for CPPInstance
    static bool CPPInstance_Check(PyObject* pyobject);
    static bool CPPInstance_CheckExact(PyObject* pyobject);

// type verifiers for CPPOverload
    static bool CPPOverload_Check(PyObject* pyobject);
    static bool CPPOverload_CheckExact(PyObject* pyobject);

// object proxy to void* conversion
    static void* CPPInstance_AsVoidPtr(PyObject* pyobject);

// void* to object proxy conversion, returns a new reference
    static PyObject* CPPInstance_FromVoidPtr(
        void* addr, const char* classname, bool python_owns = false);

    virtual ~TPython() { }
};

#endif // !CPYCPPYY_TPYTHON
