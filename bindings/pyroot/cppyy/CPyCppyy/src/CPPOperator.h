#ifndef CPYCPPYY_CPPOPERATOR_H
#define CPYCPPYY_CPPOPERATOR_H

// Bindings
#include "CPPMethod.h"

// Standard
#include <string>


namespace CPyCppyy {

class CPPOperator : public CPPMethod {
public:
    CPPOperator(Cppyy::TCppScope_t scope, Cppyy::TCppMethod_t method, const std::string& name);

public:
    PyCallable* Clone() override { return new CPPOperator(*this); }
    PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr) override;

private:
    binaryfunc fStub;
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPOPERATOR_H
