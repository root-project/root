#ifndef CPYRT_EXECUTORS_H
#define CPYRT_EXECUTORS_H

#include "Python.h"
#include "cppjit_interop.h"

// Bindings
#include "Dimensions.h"

// Standard
#include <string>

namespace cppjit::cpyrt {

struct CallContext;

class CPYRT_CLASS_EXPORT Executor {
public:
  virtual ~Executor();
  virtual PyObject* Execute(interop::TCppMethod_t, interop::TCppObject_t,
                            CallContext*) = 0;
  virtual bool HasState() { return false; }
};

// special case needed for CPPSetItem
class RefExecutor : public Executor {
public:
  RefExecutor() : fAssignable(nullptr) {}
  virtual bool SetAssignable(PyObject*);
  bool HasState() override { return true; }

protected:
  PyObject* fAssignable;
};

// create/destroy executor from fully qualified type (public API)
CPYRT_EXPORT Executor* CreateExecutor(const std::string& fullType, cdims_t = 0);
CPYRT_EXPORT Executor* CreateExecutor(interop::TCppType_t type, cdims_t = 0);
CPYRT_EXPORT void DestroyExecutor(Executor* p);
typedef Executor* (*ef_t)(cdims_t);
CPYRT_EXPORT bool RegisterExecutor(const std::string& name, ef_t fac);
CPYRT_EXPORT bool RegisterExecutorAlias(const std::string& name,
                                        const std::string& target);
CPYRT_EXPORT bool UnregisterExecutor(const std::string& name);

// helper for the actual call
CPYRT_EXPORT void* CallVoidP(interop::TCppMethod_t, interop::TCppObject_t,
                             CallContext*);

} // namespace cppjit::cpyrt

#endif // !CPYRT_EXECUTORS_H
