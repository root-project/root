#ifndef CPYCPPYY_CPPFUNCTION_H
#define CPYCPPYY_CPPFUNCTION_H

// Bindings
#include "CPPMethod.h"


namespace CPyCppyy {

class CPPFunction : public CPPMethod {
public:
    using CPPMethod::CPPMethod;

    virtual PyCallable* Clone() { return new CPPFunction(*this); }

    virtual PyObject* Call(
        CPPInstance*&, PyObject* args, PyObject* kwds, CallContext* ctx = nullptr);

protected:
    virtual PyObject* PreProcessArgs(CPPInstance*& self, PyObject* args, PyObject* kwds);
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPFUNCTION_H
