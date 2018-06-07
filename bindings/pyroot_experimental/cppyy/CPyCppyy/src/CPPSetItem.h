#ifndef CPYCPPYY_CPPSETITEM_H
#define CPYCPPYY_CPPSETITEM_H

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

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPSETITEM_H
