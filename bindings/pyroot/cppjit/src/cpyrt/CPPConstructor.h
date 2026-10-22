#ifndef CPYRT_CPPCONSTRUCTOR_H
#define CPYRT_CPPCONSTRUCTOR_H

// Bindings
#include "CPPMethod.h"

namespace cppjit::cpyrt {

class CPPConstructor : public CPPMethod {
public:
  using CPPMethod::CPPMethod;

public:
  PyObject* GetDocString() override;
  PyObject*
      Reflex(interop::Reflex::RequestId_t,
             interop::Reflex::FormatId_t = interop::Reflex::OPTIMAL) override;

  PyCallable* Clone() override { return new CPPConstructor(*this); }
  PyObject* Call(CPPInstance*& self, cpyrt_PyArgs_t args, size_t nargsf,
                 PyObject* kwds, CallContext* ctxt = nullptr) override;

protected:
  bool InitExecutor_(Executor*&, CallContext* ctxt = nullptr) override;
  bool ResultIsPyObject() const override { return false; }
};

// specialization for multiple inheritance disambiguation
class CPPMultiConstructor : public CPPConstructor {
public:
  CPPMultiConstructor(interop::TCppScope_t scope, interop::TCppMethod_t method);
  CPPMultiConstructor(const CPPMultiConstructor&);
  CPPMultiConstructor& operator=(const CPPMultiConstructor&);

public:
  PyCallable* Clone() override { return new CPPMultiConstructor(*this); }
  PyObject* Call(CPPInstance*& self, cpyrt_PyArgs_t args, size_t nargsf,
                 PyObject* kwds, CallContext* ctxt = nullptr) override;

private:
  Py_ssize_t fNumBases;
};

// specializations of prohibiting constructors
class CPPAbstractClassConstructor : public CPPConstructor {
public:
  using CPPConstructor::CPPConstructor;

public:
  PyCallable* Clone() override {
    return new CPPAbstractClassConstructor(*this);
  }
  PyObject* Call(CPPInstance*& self, cpyrt_PyArgs_t args, size_t nargsf,
                 PyObject* kwds, CallContext* ctxt = nullptr) override;
};

class CPPNamespaceConstructor : public CPPConstructor {
public:
  using CPPConstructor::CPPConstructor;

public:
  PyCallable* Clone() override { return new CPPNamespaceConstructor(*this); }
  PyObject* Call(CPPInstance*& self, cpyrt_PyArgs_t args, size_t nargsf,
                 PyObject* kwds, CallContext* ctxt = nullptr) override;
};

class CPPIncompleteClassConstructor : public CPPConstructor {
public:
  using CPPConstructor::CPPConstructor;

public:
  PyCallable* Clone() override {
    return new CPPIncompleteClassConstructor(*this);
  }
  PyObject* Call(CPPInstance*& self, cpyrt_PyArgs_t args, size_t nargsf,
                 PyObject* kwds, CallContext* ctxt = nullptr) override;
};

class CPPAllPrivateClassConstructor : public CPPConstructor {
public:
  using CPPConstructor::CPPConstructor;

public:
  PyCallable* Clone() override {
    return new CPPAllPrivateClassConstructor(*this);
  }
  PyObject* Call(CPPInstance*& self, cpyrt_PyArgs_t args, size_t nargsf,
                 PyObject* kwds, CallContext* ctxt = nullptr) override;
};

} // namespace cppjit::cpyrt

#endif // !CPYRT_CPPCONSTRUCTOR_H
