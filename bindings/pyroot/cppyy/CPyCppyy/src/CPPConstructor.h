#ifndef CPYCPPYY_CPPCONSTRUCTOR_H
#define CPYCPPYY_CPPCONSTRUCTOR_H

// Bindings
#include "CPPMethod.h"


namespace CPyCppyy {

class CPPConstructor : public CPPMethod {
public:
    using CPPMethod::CPPMethod;

public:
    PyObject* GetDocString() override;
    PyObject* Reflex(Cppyy::Reflex::RequestId_t,
                             Cppyy::Reflex::FormatId_t = Cppyy::Reflex::OPTIMAL) override;

    PyCallable* Clone() override { return new CPPConstructor(*this); }
    PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr) override;

protected:
    bool InitExecutor_(Executor*&, CallContext* ctxt = nullptr) override;
};


// specialization for multiple inheritance disambiguation
class CPPMultiConstructor : public CPPConstructor {
public:
    CPPMultiConstructor(Cppyy::TCppScope_t scope, Cppyy::TCppMethod_t method);
    CPPMultiConstructor(const CPPMultiConstructor&);
    CPPMultiConstructor& operator=(const CPPMultiConstructor&);

public:
    PyCallable* Clone() override { return new CPPMultiConstructor(*this); }
    PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr) override;

private:
    Py_ssize_t fNumBases;
};


// specializations of prohibiting constructors
class CPPAbstractClassConstructor : public CPPConstructor {
public:
    using CPPConstructor::CPPConstructor;

public:
    PyCallable* Clone() override { return new CPPAbstractClassConstructor(*this); }
    PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr) override;
};

class CPPNamespaceConstructor : public CPPConstructor {
public:
    using CPPConstructor::CPPConstructor;

public:
    PyCallable* Clone() override { return new CPPNamespaceConstructor(*this); }
    PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr) override;
};

class CPPIncompleteClassConstructor : public CPPConstructor {
public:
    using CPPConstructor::CPPConstructor;

public:
    PyCallable* Clone() override { return new CPPIncompleteClassConstructor(*this); }
    PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr) override;
};

class CPPAllPrivateClassConstructor : public CPPConstructor {
public:
    using CPPConstructor::CPPConstructor;

public:
    PyCallable* Clone() override { return new CPPAllPrivateClassConstructor(*this); }
    PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr) override;
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPCONSTRUCTOR_H
