#ifndef CPYCPPYY_PYCALLABLE_H
#define CPYCPPYY_PYCALLABLE_H

#include <climits>

// Bindings
#include "CPyCppyy/Reflex.h"
#include "CallContext.h"


namespace CPyCppyy {

class CPPInstance;

class PyCallable {
public:
    virtual ~PyCallable() {}

public:
    virtual PyObject* GetSignature(bool show_formalargs = true) = 0;
    virtual PyObject* GetSignatureNames() = 0;
    virtual PyObject* GetSignatureTypes() = 0;
    virtual PyObject* GetPrototype(bool show_formalargs = true) = 0;
    virtual PyObject* GetTypeName() { return GetPrototype(false); }
    virtual PyObject* GetDocString() { return GetPrototype(); }
    virtual PyObject* Reflex(Cppyy::Reflex::RequestId_t request,
                             Cppyy::Reflex::FormatId_t format = Cppyy::Reflex::OPTIMAL) {
        PyErr_Format(PyExc_ValueError, "unsupported reflex request %d or format %d", request, format);
        return nullptr;
    };

    virtual int GetPriority() = 0;
    virtual bool IsGreedy() = 0;

    virtual int GetMaxArgs() = 0;
    virtual PyObject* GetCoVarNames() = 0;
    virtual PyObject* GetArgDefault(int /* iarg */, bool silent=true) = 0;
    virtual bool IsConst() { return false; }

    virtual PyObject* GetScopeProxy() = 0;
    virtual Cppyy::TCppFuncAddr_t GetFunctionAddress() = 0;

    virtual PyCallable* Clone() = 0;

    virtual int GetArgMatchScore(PyObject* /* args_tuple */) { return INT_MAX; }

public:
    virtual PyObject* Call(CPPInstance*& self,
        CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt = nullptr) = 0;
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_PYCALLABLE_H
