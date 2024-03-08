#ifndef CPYCPPYY_CPPCLASSMETHOD_H
#define CPYCPPYY_CPPCLASSMETHOD_H

// Bindings
#include "CPPMethod.h"


namespace CPyCppyy {

class CPPClassMethod : public CPPMethod {
public:
    using CPPMethod::CPPMethod;

    virtual PyCallable* Clone() { return new CPPClassMethod(*this); }
    virtual PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr);
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPCLASSMETHOD_H
