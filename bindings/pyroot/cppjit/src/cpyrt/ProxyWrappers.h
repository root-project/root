#ifndef CPYRT_PROXYWRAPPERS_H
#define CPYRT_PROXYWRAPPERS_H

// Bindings
#include "Dimensions.h"

// Standard
#include <string>

namespace cppjit::cpyrt {

// construct a Python shadow class for the named C++ class
PyObject* GetScopeProxy(interop::TCppScope_t);
PyObject* CreateScopeProxy(PyObject*, PyObject* args);
PyObject* CreateScopeProxy(const std::string& scope_name,
                           PyObject* parent = nullptr,
                           const unsigned flags = 0);

PyObject* CreateScopeProxy(interop::TCppScope_t scope,
                           PyObject* parent = nullptr,
                           const unsigned flags = 0);
// C++ exceptions form a special case b/c they have to derive from BaseException
PyObject* CreateExcScopeProxy(PyObject* pyscope, PyObject* pyname,
                              PyObject* parent);

// bind a C++ object into a Python proxy object (flags are CPPInstance::Default)
PyObject* BindCppObjectNoCast(interop::TCppObject_t object,
                              interop::TCppScope_t klass,
                              const unsigned flags = 0);
PyObject* BindCppObject(interop::TCppObject_t object,
                        interop::TCppScope_t klass, const unsigned flags = 0);
PyObject* BindCppObjectArray(interop::TCppObject_t address,
                             interop::TCppScope_t klass, cdims_t dims);

} // namespace cppjit::cpyrt

#endif // !CPYRT_PROXYWRAPPERS_H
