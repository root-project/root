#ifndef CPYRT_CPPCLASSMETHOD_H
#define CPYRT_CPPCLASSMETHOD_H

// Bindings
#include "CPPMethod.h"

namespace cppjit::cpyrt {

class CPPClassMethod : public CPPMethod {
public:
  using CPPMethod::CPPMethod;

public:
  PyObject* GetTypeName() override;

public:
  PyCallable* Clone() override { return new CPPClassMethod(*this); }
  PyObject* Call(CPPInstance*& self, cpyrt_PyArgs_t args, size_t nargsf,
                 PyObject* kwds, CallContext* ctxt = nullptr) override;
};

} // namespace cppjit::cpyrt

#endif // !CPYRT_CPPCLASSMETHOD_H
