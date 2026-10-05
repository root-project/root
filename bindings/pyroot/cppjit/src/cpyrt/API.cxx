// Bindings
#include "cpyrt.h"

using namespace cppjit;
#include "cppjit_interop.h"
#define CPYRT_INTERNAL 1
#include "cpyrt/API.h"
#undef CPYRT_INTERNAL

#include "CPPInstance.h"
#include "CPPOverload.h"
#include "CPPScope.h"
#include "ProxyWrappers.h"
#include "cpyrt/DispatchPtr.h"

// Standard
#include <string>

//______________________________________________________________________________
//                  cpyrt API: Interpreter and Proxy Access
//                  ==========================================
//
// Access to cppjit Python objects from Cling and C++: allows conversion for
// instances and type checking for scopes, instances, etc.

namespace cppjit::cpyrt {
extern PyObject* gThisModule;
}

//- private helpers ----------------------------------------------------------
namespace {

static bool Initialize() {
  // Private initialization method: load the cppjit module. Expects the Python
  // interpreter to be already initialized.
  static bool isInitialized = false;
  if (isInitialized)
    return true;

  // cppjit is a Python extension library, not an embedding library: the
  // Python interpreter must be initialized by the embedding application
  // (e.g. TPython) before any cpyrt API function is called.
  if (!Py_IsInitialized())
    return false;

  // Importing the extension module is what runs the cpyrt initialization
  // that sets gThisModule.
  if (!cpyrt::gThisModule) {
    PyObject* cppjitmod = PyImport_ImportModule("cppjit");
    if (!cppjitmod)
      return false;
    Py_DECREF(cppjitmod);
  }

  // declare success ...
  isInitialized = true;
  return true;
}

} // unnamed namespace

//- C++ access to cppjit objects ---------------------------------------------
std::string cpyrt::Instance_GetScopedFinalName(PyObject* pyobject) {
  if (!Instance_Check(pyobject)) {
    PyErr_SetString(
        PyExc_TypeError,
        "Instance_GetScopedFinalName : object is not a C++ instance");
    return "";
  }

  interop::TCppScope_t pyobjectClass = ((CPPInstance*)pyobject)->ObjectIsA();
  return interop::GetScopedFinalName(pyobjectClass);
}

//-----------------------------------------------------------------------------
void* cpyrt::Instance_AsVoidPtr(PyObject* pyobject) {
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
PyObject* cpyrt::Instance_FromVoidPtr(void* addr, const std::string& classname,
                                      bool python_owns) {
  // Bind the addr to a python object of class defined by classname.
  if (!Initialize())
    return nullptr;

  PythonGILRAII python_gil_raii;

  // perform cast (the call will check TClass and addr, and set python errors)
  PyObject* pyobject =
      BindCppObjectNoCast(addr, interop::GetScope(classname), false);

  // give ownership, for ref-counting, to the python side, if so requested
  if (python_owns && CPPInstance_Check(pyobject))
    ((CPPInstance*)pyobject)->PythonOwns();

  return pyobject;
}

//-----------------------------------------------------------------------------
PyObject* cpyrt::Instance_FromVoidPtr(void* addr,
                                      interop::TCppScope_t klass_scope,
                                      bool python_owns) {
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
namespace cppjit::cpyrt {
// version with C type arguments only for use with Numba
PyObject* Instance_FromVoidPtr(void* addr, const char* classname,
                               int python_owns) {
  return Instance_FromVoidPtr(addr, std::string(classname), (bool)python_owns);
}
} // namespace cppjit::cpyrt

//-----------------------------------------------------------------------------
bool cpyrt::Scope_Check(PyObject* pyobject) {
  // Test if the given object is of a CPPScope derived type.
  if (!Initialize())
    return false;

  PythonGILRAII python_gil_raii;
  return CPPScope_Check(pyobject);
}

//-----------------------------------------------------------------------------
bool cpyrt::Scope_CheckExact(PyObject* pyobject) {
  // Test if the given object is of a CPPScope type.
  if (!Initialize())
    return false;

  PythonGILRAII python_gil_raii;
  return CPPScope_CheckExact(pyobject);
}

//-----------------------------------------------------------------------------
bool cpyrt::Instance_Check(PyObject* pyobject) {
  // Test if the given pyobject is of CPPInstance derived type.
  if (!Initialize())
    return false;

  PythonGILRAII python_gil_raii;
  // detailed walk through inheritance hierarchy
  return CPPInstance_Check(pyobject);
}

//-----------------------------------------------------------------------------
bool cpyrt::Instance_CheckExact(PyObject* pyobject) {
  // Test if the given pyobject is of CPPInstance type.
  if (!Initialize())
    return false;

  PythonGILRAII python_gil_raii;
  // direct pointer comparison of type member
  return CPPInstance_CheckExact(pyobject);
}

//-----------------------------------------------------------------------------
void cpyrt::Instance_SetPythonOwns(PyObject* pyobject) {
  if (!Initialize())
    return;

  // check validity of cast
  if (!CPPInstance_Check(pyobject))
    return;

  ((CPPInstance*)pyobject)->PythonOwns();
}

//-----------------------------------------------------------------------------
void cpyrt::Instance_SetCppOwns(PyObject* pyobject) {
  if (!Initialize())
    return;

  // check validity of cast
  if (!CPPInstance_Check(pyobject))
    return;

  ((CPPInstance*)pyobject)->CppOwns();
}

//-----------------------------------------------------------------------------
bool cpyrt::Sequence_Check(PyObject* pyobject) {
  PythonGILRAII python_gil_raii;
  // Extends on PySequence_Check() to determine whether an object can be
  // iterated over (technically, all objects can b/c of C++ pointer arithmetic,
  // hence this check isn't 100% accurate, but neither is PySequence_Check()).

  // Note: simply having the iterator protocol does not constitute a sequence,
  // bc PySequence_GetItem() would fail.

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
bool cppjit::cpyrt::Instance_IsLively(PyObject* pyobject) {
  PythonGILRAII python_gil_raii;
  // Test whether the given instance can safely return to C++
  if (!CPPInstance_Check(pyobject))
    return true; // simply don't know

  // the instance fails the lively test if it owns the C++ object while having a
  // reference count of 1 (meaning: it could delete the C++ instance any moment)
  if (Py_REFCNT(pyobject) <= 1 &&
      (((CPPInstance*)pyobject)->fFlags & CPPInstance::kIsOwner))
    return false;

  return true;
}

//-----------------------------------------------------------------------------
bool cpyrt::Overload_Check(PyObject* pyobject) {
  // Test if the given pyobject is of CPPOverload derived type.
  if (!Initialize())
    return false;

  PythonGILRAII python_gil_raii;
  // detailed walk through inheritance hierarchy
  return CPPOverload_Check(pyobject);
}

//-----------------------------------------------------------------------------
bool cpyrt::Overload_CheckExact(PyObject* pyobject) {
  // Test if the given pyobject is of CPPOverload type.
  if (!Initialize())
    return false;

  PythonGILRAII python_gil_raii;
  // direct pointer comparison of type member
  return CPPOverload_CheckExact(pyobject);
}

//-----------------------------------------------------------------------------
void cpyrt::Instance_SetReduceMethod(PyCFunction reduceMethod) {
  CPPInstance::ReduceMethod() = reduceMethod;
}

//-----------------------------------------------------------------------------
PyObject* cpyrt::GetThisModule() {
  // Return the cppjit extension module as a borrowed reference. Frameworks
  // embedding Python (e.g. TPython) can use it to attach imported python
  // modules, so that no python proxy is created for the C++ proxy.
  if (!Initialize())
    return nullptr;
  return gThisModule;
}
