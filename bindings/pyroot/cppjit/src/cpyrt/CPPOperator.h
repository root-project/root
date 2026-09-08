#ifndef CPYRT_CPPOPERATOR_H
#define CPYRT_CPPOPERATOR_H

// Bindings
#include "CPPMethod.h"

// Standard
#include <string>

namespace cppjit::cpyrt {

class CPPOperator : public CPPMethod {
public:
  CPPOperator(interop::TCppScope_t scope, interop::TCppMethod_t method,
              const std::string& name);

public:
  PyCallable* Clone() override { return new CPPOperator(*this); }
  PyObject* Call(CPPInstance*& self, cpyrt_PyArgs_t args, size_t nargsf,
                 PyObject* kwds, CallContext* ctxt = nullptr) override;

private:
  binaryfunc fStub;
};

} // namespace cppjit::cpyrt

#endif // !CPYRT_CPPOPERATOR_H
