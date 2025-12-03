#ifndef CPYCPPYY_CPPFUNCTION_H
#define CPYCPPYY_CPPFUNCTION_H

// Bindings
#include "CPPMethod.h"


namespace CPyCppyy {

// Wrapper for global free/static C++ functions
class CPPFunction : public CPPMethod {
public:
    using CPPMethod::CPPMethod;

public:
    PyObject* GetTypeName() override;

public:
    PyCallable* Clone() override { return new CPPFunction(*this); }
    PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr) override;

protected:
    bool ProcessArgs(PyCallArgs& args) override;
};

// Wrapper for global binary operators that swap arguments
class CPPReverseBinary : public CPPFunction {
public:
    using CPPFunction::CPPFunction;

    PyCallable* Clone() override { return new CPPFunction(*this); }
    PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr) override;

protected:
    bool ProcessArgs(PyCallArgs& args) override;
};

// Helper to add self to the arguments tuple if rebound
bool AdjustSelf(PyCallArgs& cargs);

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPFUNCTION_H
