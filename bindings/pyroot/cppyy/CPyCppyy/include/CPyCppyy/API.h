#ifndef CPYCPPYY_API_H
#define CPYCPPYY_API_H

//
// Access to the python interpreter and API onto CPyCppyy.
//

// Python
#ifdef _WIN32
#pragma warning (disable : 4275)
#pragma warning (disable : 4251)
#pragma warning (disable : 4800)
#endif
#if defined(linux)
#include <stdio.h>
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#ifdef _FILE_OFFSET_BITS
#undef _FILE_OFFSET_BITS
#endif
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif
#endif
#include "Python.h"

#define CPYCPPYY_VERSION_HEX 0x010c10

// Cppyy types
namespace Cppyy {
    typedef size_t      TCppScope_t;
    typedef TCppScope_t TCppType_t;
    typedef void*       TCppEnum_t;
    typedef void*       TCppObject_t;
    typedef intptr_t    TCppMethod_t;

    typedef size_t      TCppIndex_t;
    typedef void*       TCppFuncAddr_t;
} // namespace Cppyy

// Bindings
#include "CPyCppyy/PyResult.h"
#include "CPyCppyy/CommonDefs.h"

// Standard
#include <string>
#include <vector>


namespace CPyCppyy {

//- type conversion ---------------------------------------------------------

#ifndef CPYCPPYY_PARAMETER
#define CPYCPPYY_PARAMETER
// generic function argument type
struct Parameter {
    union Value {
        bool                 fBool;
        int8_t               fInt8;
        uint8_t              fUInt8;
        short                fShort;
        unsigned short       fUShort;
        int                  fInt;
        unsigned int         fUInt;
        long                 fLong;
        intptr_t             fIntPtr;
        unsigned long        fULong;
        long long            fLLong;
        unsigned long long   fULLong;
        int64_t              fInt64;
        uint64_t             fUInt64;
        float                fFloat;
        double               fDouble;
        long double          fLDouble;
        void*                fVoidp;
    } fValue;
    void* fRef;
    char  fTypeCode;
};
#endif // CPYCPPYY_PARAMETER

// CallContext is not currently exposed
struct CallContext;

// Dimensions class not currently exposed
#ifndef CPYCPPYY_DIMENSIONS_H
#define CPYCPPYY_DIMENSIONS_H
typedef Py_ssize_t dim_t;

class Dimensions {      // Windows note: NOT exported/imported
    dim_t* fDims;

public:
    Dimensions(dim_t /*ndim*/ = 0, dim_t* /*dims*/ = nullptr) : fDims(nullptr) {}
    ~Dimensions() { delete [] fDims; }

public:
    operator bool() const { return (bool)fDims; }
};

typedef Dimensions dims_t;
typedef const dims_t& cdims_t;
#endif // !CPYCPPYY_DIMENSIONS_H

// type converter base class
class CPYCPPYY_CLASS_EXTERN Converter {
public:
    virtual ~Converter();

// convert the python object and add store it on the parameter
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr) = 0;

// convert a C++ object from memory to a Python object
    virtual PyObject* FromMemory(void* address);

// convert a Python object to a C++ object and store it on address
    virtual bool ToMemory(PyObject* value, void* address, PyObject* ctxt = nullptr);

// if a converter has state, it will be unique per function, shared otherwise
    virtual bool HasState() { return false; }
};

// create a converter based on its full type name and dimensions
CPYCPPYY_EXTERN Converter* CreateConverter(const std::string& name, cdims_t = 0);

// delete a previously created converter
CPYCPPYY_EXTERN void DestroyConverter(Converter* p);

// register a custom converter
typedef Converter* (*ConverterFactory_t)(cdims_t);
CPYCPPYY_EXTERN bool RegisterConverter(const std::string& name, ConverterFactory_t);

// register a custom converter that is a reference to an existing converter
CPYCPPYY_EXTERN bool RegisterConverterAlias(const std::string& name, const std::string& target);

// remove a custom converter
CPYCPPYY_EXTERN bool UnregisterConverter(const std::string& name);


// function executor base class
class CPYCPPYY_CLASS_EXTERN Executor {
public:
    virtual ~Executor();

// callback when executing a function from Python
    virtual PyObject* Execute(
        Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*) = 0;

// if an executor has state, it will be unique per function, shared otherwise
    virtual bool HasState() { return false; }
};

// create an executor based on its full type name
CPYCPPYY_EXTERN Executor* CreateExecutor(const std::string& name, cdims_t = 0);

// delete a previously created executor
CPYCPPYY_EXTERN void DestroyConverter(Converter* p);

// register a custom executor
typedef Executor* (*ExecutorFactory_t)(cdims_t);
CPYCPPYY_EXTERN bool RegisterExecutor(const std::string& name, ExecutorFactory_t);

// register a custom executor that is a reference to an existing converter
CPYCPPYY_EXTERN bool RegisterExecutorAlias(const std::string& name, const std::string& target);

// remove a custom executor
CPYCPPYY_EXTERN bool UnregisterExecutor(const std::string& name);

// helper for calling into C++ from a custom executor
CPYCPPYY_EXTERN void* CallVoidP(Cppyy::TCppMethod_t, Cppyy::TCppObject_t, CallContext*);


//- C++ access to cppyy objects ---------------------------------------------

// C++ Instance (python object proxy) to void* conversion
CPYCPPYY_EXTERN void* Instance_AsVoidPtr(PyObject* pyobject);

// void* to C++ Instance (python object proxy) conversion, returns a new reference
CPYCPPYY_EXTERN PyObject* Instance_FromVoidPtr(
    void* addr, const std::string& classname, bool python_owns = false);

// type verifiers for C++ Scope
CPYCPPYY_EXTERN bool Scope_Check(PyObject* pyobject);
CPYCPPYY_EXTERN bool Scope_CheckExact(PyObject* pyobject);

// type verifiers for C++ Instance
CPYCPPYY_EXTERN bool Instance_Check(PyObject* pyobject);
CPYCPPYY_EXTERN bool Instance_CheckExact(PyObject* pyobject);

// type verifier for sequences
CPYCPPYY_EXTERN bool Sequence_Check(PyObject* pyobject);

// helper to verify expected safety of moving an instance into C++
CPYCPPYY_EXTERN bool Instance_IsLively(PyObject* pyobject);

// type verifiers for C++ Overload
CPYCPPYY_EXTERN bool Overload_Check(PyObject* pyobject);
CPYCPPYY_EXTERN bool Overload_CheckExact(PyObject* pyobject);


//- access to the python interpreter ----------------------------------------

// import a python module, making its classes available to Cling
CPYCPPYY_EXTERN bool Import(const std::string& name);

// execute a python statement (e.g. "import sys")
CPYCPPYY_EXTERN bool Exec(const std::string& cmd);

// evaluate a python expression (e.g. "1+1")
CPYCPPYY_EXTERN const PyResult Eval(const std::string& expr);

// execute a python stand-alone script, with argv CLI arguments
CPYCPPYY_EXTERN void ExecScript(const std::string& name, const std::vector<std::string>& args);

// enter an interactive python session (exit with ^D)
CPYCPPYY_EXTERN void Prompt();

} // namespace CPyCppyy

#endif // !CPYCPPYY_API_H
