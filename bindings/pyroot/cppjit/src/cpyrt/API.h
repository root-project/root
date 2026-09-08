#ifndef CPYRT_API_H
#define CPYRT_API_H

//
// Access to the python interpreter and API onto cpyrt.
//

// Python
#ifdef _WIN32
#pragma warning(disable : 4275)
#pragma warning(disable : 4251)
#pragma warning(disable : 4800)
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

#define CPYRT_VERSION_HEX 0x011200

// cppjit_interop types
#ifndef CPYRT_INTERNAL

namespace Cpp {
struct DeclRef;
struct TypeRef;
struct FuncRef;
struct ObjectRef;
} // namespace Cpp

namespace cppjit::interop {
typedef Cpp::DeclRef TCppScope_t;
typedef Cpp::TypeRef TCppType_t;
typedef Cpp::ObjectRef TCppObject_t;
typedef Cpp::FuncRef TCppMethod_t;
typedef size_t TCppIndex_t;
typedef void* TCppFuncAddr_t;
} // namespace cppjit::interop
#endif

// Bindings
#include "cpyrt/CommonDefs.h"

// Standard
#include <string>
#include <vector>

namespace cppjit::cpyrt {

//- type conversion ---------------------------------------------------------

#ifndef CPYRT_PARAMETER
#define CPYRT_PARAMETER
// generic function argument type
struct Parameter {
  union Value {
    bool fBool;
    int8_t fInt8;
    uint8_t fUInt8;
    short fShort;
    unsigned short fUShort;
    int fInt;
    unsigned int fUInt;
    long fLong;
    intptr_t fIntPtr;
    unsigned long fULong;
    long long fLLong;
    unsigned long long fULLong;
    int64_t fInt64;
    uint64_t fUInt64;
    float fFloat;
    double fDouble;
    long double fLDouble;
    void* fVoidp;
  } fValue;
  void* fRef;
  char fTypeCode;
};
#endif // CPYRT_PARAMETER

// CallContext is not currently exposed
struct CallContext;

// Dimensions class not currently exposed
#ifndef CPYRT_DIMENSIONS_H
#define CPYRT_DIMENSIONS_H
typedef Py_ssize_t dim_t;

class Dimensions { // Windows note: NOT exported/imported
  dim_t* fDims;

public:
  Dimensions(dim_t /*ndim*/ = 0, dim_t* /*dims*/ = nullptr) : fDims(nullptr) {}
  ~Dimensions() { delete[] fDims; }

public:
  operator bool() const { return (bool)fDims; }
};

typedef Dimensions dims_t;
typedef const dims_t& cdims_t;
#endif // !CPYRT_DIMENSIONS_H

// type converter base class
class CPYRT_CLASS_EXTERN Converter {
public:
  virtual ~Converter();

  // convert the python object and add store it on the parameter
  virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr) = 0;

  // convert a C++ object from memory to a Python object
  virtual PyObject* FromMemory(void* address);

  // convert a Python object to a C++ object and store it on address
  virtual bool ToMemory(PyObject* value, void* address,
                        PyObject* ctxt = nullptr);

  // if a converter has state, it will be unique per function, shared otherwise
  virtual bool HasState() { return false; }
};

// create a converter based on its full type name and dimensions
CPYRT_EXTERN Converter* CreateConverter(const std::string& name, cdims_t = 0);
CPYRT_EXTERN Converter* CreateConverter(interop::TCppType_t type, cdims_t = 0);

// delete a previously created converter
CPYRT_EXTERN void DestroyConverter(Converter* p);

// register a custom converter
typedef Converter* (*ConverterFactory_t)(cdims_t);
CPYRT_EXTERN bool RegisterConverter(const std::string& name,
                                    ConverterFactory_t);

// register a custom converter that is a reference to an existing converter
CPYRT_EXTERN bool RegisterConverterAlias(const std::string& name,
                                         const std::string& target);

// remove a custom converter
CPYRT_EXTERN bool UnregisterConverter(const std::string& name);

// function executor base class
class CPYRT_CLASS_EXTERN Executor {
public:
  virtual ~Executor();

  // callback when executing a function from Python
  virtual PyObject* Execute(interop::TCppMethod_t, interop::TCppObject_t,
                            CallContext*) = 0;

  // if an executor has state, it will be unique per function, shared otherwise
  virtual bool HasState() { return false; }
};

// create an executor based on its full type name
CPYRT_EXTERN Executor* CreateExecutor(const std::string& name, cdims_t = 0);
CPYRT_EXTERN Executor* CreateExecutor(interop::TCppType_t type, cdims_t = 0);

// delete a previously created executor
CPYRT_EXTERN void DestroyConverter(Converter* p);

// register a custom executor
typedef Executor* (*ExecutorFactory_t)(cdims_t);
CPYRT_EXTERN bool RegisterExecutor(const std::string& name, ExecutorFactory_t);

// register a custom executor that is a reference to an existing converter
CPYRT_EXTERN bool RegisterExecutorAlias(const std::string& name,
                                        const std::string& target);

// remove a custom executor
CPYRT_EXTERN bool UnregisterExecutor(const std::string& name);

// helper for calling into C++ from a custom executor
CPYRT_EXTERN void* CallVoidP(interop::TCppMethod_t, interop::TCppObject_t,
                             CallContext*);

//- C++ access to cppjit objects ---------------------------------------------

// Get C++ Instance (python object proxy) name.
// Sets a TypeError and returns an empty string if the pyobject is not a
// CPPInstance.
CPYRT_EXTERN std::string Instance_GetScopedFinalName(PyObject* pyobject);

// C++ Instance (python object proxy) to void* conversion
CPYRT_EXTERN void* Instance_AsVoidPtr(PyObject* pyobject);

// void* to C++ Instance (python object proxy) conversion, returns a new
// reference
CPYRT_EXTERN PyObject* Instance_FromVoidPtr(void* addr,
                                            const std::string& classname,
                                            bool python_owns = false);
CPYRT_EXTERN PyObject* Instance_FromVoidPtr(void* addr,
                                            interop::TCppScope_t klass_scope,
                                            bool python_owns = false);
// type verifiers for C++ Scope
CPYRT_EXTERN bool Scope_Check(PyObject* pyobject);
CPYRT_EXTERN bool Scope_CheckExact(PyObject* pyobject);

// type verifiers for C++ Instance
CPYRT_EXTERN bool Instance_Check(PyObject* pyobject);
CPYRT_EXTERN bool Instance_CheckExact(PyObject* pyobject);

// memory management: ownership of the underlying C++ object
CPYRT_EXTERN void Instance_SetPythonOwns(PyObject* pyobject);
CPYRT_EXTERN void Instance_SetCppOwns(PyObject* pyobject);

// type verifier for sequences
CPYRT_EXTERN bool Sequence_Check(PyObject* pyobject);

// helper to verify expected safety of moving an instance into C++
CPYRT_EXTERN bool Instance_IsLively(PyObject* pyobject);

// type verifiers for C++ Overload
CPYRT_EXTERN bool Overload_Check(PyObject* pyobject);
CPYRT_EXTERN bool Overload_CheckExact(PyObject* pyobject);

// Sets the __reduce__ method for the CPPInstance class, which is by default not
// implemented by cppjit but might make sense to implement by frameworks that
// support IO of arbitrary C++ objects, like ROOT.
CPYRT_EXTERN void Instance_SetReduceMethod(PyCFunction reduceMethod);

//- access to the python interpreter ----------------------------------------

// import a python module, making its classes available to Cling
CPYRT_EXTERN bool Import(const std::string& name);

// execute a python statement (e.g. "import sys")
CPYRT_EXTERN bool Exec(const std::string& cmd);

// execute a python stand-alone script, with argv CLI arguments
CPYRT_EXTERN void ExecScript(const std::string& name,
                             const std::vector<std::string>& args);

// enter an interactive python session (exit with ^D)
CPYRT_EXTERN void Prompt();

} // namespace cppjit::cpyrt

#endif // !CPYRT_API_H
