#ifndef CPYCPPYY_CPPCONSTRUCTOR_H
#define CPYCPPYY_CPPCONSTRUCTOR_H

// Bindings
#include "CPPMethod.h"


namespace CPyCppyy {

class CPPConstructor : public CPPMethod {
public:
    using CPPMethod::CPPMethod;

public:
    virtual PyObject* GetDocString();
    virtual PyObject* Reflex(Cppyy::Reflex::RequestId_t,
                             Cppyy::Reflex::FormatId_t = Cppyy::Reflex::OPTIMAL);

    virtual PyCallable* Clone() { return new CPPConstructor(*this); }
    virtual PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr);

protected:
    virtual bool InitExecutor_(Executor*&, CallContext* ctxt = nullptr);
};


// specialization for multiple inheritance disambiguation
class CPPMultiConstructor : public CPPConstructor {
public:
    CPPMultiConstructor(Cppyy::TCppScope_t scope, Cppyy::TCppMethod_t method);
    CPPMultiConstructor(const CPPMultiConstructor&);
    CPPMultiConstructor& operator=(const CPPMultiConstructor&);

public:
    virtual PyCallable* Clone() { return new CPPMultiConstructor(*this); }
    virtual PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr);

private:
    Py_ssize_t fNumBases;
};


// specializations of prohibiting constructors
class CPPAbstractClassConstructor : public CPPConstructor {
public:
    using CPPConstructor::CPPConstructor;

public:
    virtual PyCallable* Clone() { return new CPPAbstractClassConstructor(*this); }
    virtual PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr);
};

class CPPNamespaceConstructor : public CPPConstructor {
public:
    using CPPConstructor::CPPConstructor;

public:
    virtual PyCallable* Clone() { return new CPPNamespaceConstructor(*this); }
    virtual PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr);
};

class CPPIncompleteClassConstructor : public CPPConstructor {
public:
    using CPPConstructor::CPPConstructor;

public:
    virtual PyCallable* Clone() { return new CPPIncompleteClassConstructor(*this); }
    virtual PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr);
};

class CPPAllPrivateClassConstructor : public CPPConstructor {
public:
    using CPPConstructor::CPPConstructor;

public:
    virtual PyCallable* Clone() { return new CPPAllPrivateClassConstructor(*this); }
    virtual PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr);
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPCONSTRUCTOR_H
