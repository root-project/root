#ifndef CPYCPPYY_CPPFUNCTION_H
#define CPYCPPYY_CPPFUNCTION_H

// Bindings
#include "CPPMethod.h"


namespace CPyCppyy {

// Wrapper for global free/static C++ functions
class CPPFunction : public CPPMethod {
public:
    using CPPMethod::CPPMethod;

    virtual PyCallable* Clone() { return new CPPFunction(*this); }
    virtual PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr);

protected:
    virtual bool ProcessArgs(PyCallArgs& args);
};

// Wrapper for global binary operators that swap arguments
class CPPReverseBinary : public CPPFunction {
public:
    using CPPFunction::CPPFunction;

    virtual PyCallable* Clone() { return new CPPFunction(*this); }
    virtual PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr);

protected:
    virtual bool ProcessArgs(PyCallArgs& args);
};

// Helper to add self to the arguments tuple if rebound
bool AdjustSelf(PyCallArgs& cargs);

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPFUNCTION_H
