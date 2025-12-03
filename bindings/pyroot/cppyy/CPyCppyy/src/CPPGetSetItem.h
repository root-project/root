#ifndef CPYCPPYY_CPPGETSETITEM_H
#define CPYCPPYY_CPPGETSETITEM_H

// Bindings
#include "CPPMethod.h"


namespace CPyCppyy {

class CPPSetItem : public CPPMethod {
public:
    using CPPMethod::CPPMethod;

public:
    PyCallable* Clone() override { return new CPPSetItem(*this); }

protected:
    bool ProcessArgs(PyCallArgs& args) override;
    bool InitExecutor_(Executor*&, CallContext* ctxt = nullptr) override;
};

class CPPGetItem : public CPPMethod {
public:
    using CPPMethod::CPPMethod;

public:
    PyCallable* Clone() override { return new CPPGetItem(*this); }

protected:
    bool ProcessArgs(PyCallArgs& args) override;
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPGETSETITEM_H
