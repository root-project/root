#ifndef CPYCPPYY_CPPCLASSMETHOD_H
#define CPYCPPYY_CPPCLASSMETHOD_H

// Bindings
#include "CPPMethod.h"


namespace CPyCppyy {

class CPPClassMethod : public CPPMethod {
public:
    using CPPMethod::CPPMethod;

public:
    PyObject* GetTypeName() override;

public:
    PyCallable* Clone() override { return new CPPClassMethod(*this); }
    PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr) override;
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPCLASSMETHOD_H
