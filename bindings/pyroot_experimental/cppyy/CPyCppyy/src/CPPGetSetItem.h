#ifndef CPYCPPYY_CPPGETSETITEM_H
#define CPYCPPYY_CPPGETSETITEM_H

// Bindings
#include "CPPMethod.h"


namespace CPyCppyy {

class CPPSetItem : public CPPMethod {
public:
    using CPPMethod::CPPMethod;

public:
    virtual PyCallable* Clone() { return new CPPSetItem(*this); }
    virtual PyObject* PreProcessArgs(CPPInstance*& self, PyObject* args, PyObject* kwds);

protected:
    virtual bool InitExecutor_(Executor*&, CallContext* ctxt = nullptr);
};

class CPPGetItem : public CPPMethod {
public:
    using CPPMethod::CPPMethod;

public:
    virtual PyCallable* Clone() { return new CPPGetItem(*this); }
    virtual PyObject* PreProcessArgs(CPPInstance*& self, PyObject* args, PyObject* kwds);
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPGETSETITEM_H
