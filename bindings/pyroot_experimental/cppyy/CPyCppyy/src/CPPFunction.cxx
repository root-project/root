// Bindings
#include "CPyCppyy.h"
#include "CPPFunction.h"
#include "CPPInstance.h"


//- CPPFunction public members --------------------------------------------------
PyObject* CPyCppyy::CPPFunction::PreProcessArgs(
        CPPInstance*& self, PyObject* args, PyObject*)
{
// add self as part of the function arguments (means bound member)
    Py_ssize_t sz = PyTuple_GET_SIZE(args);
    PyObject* newArgs = PyTuple_New(sz+1);
    for (int i = 0; i < sz; ++i) {
        PyObject* item = PyTuple_GET_ITEM(args, i);
        Py_INCREF(item);
        PyTuple_SET_ITEM(newArgs, i+1, item);
    }

    Py_INCREF(self);
    PyTuple_SET_ITEM(newArgs, 0, (PyObject*)self);

    return newArgs;
}

//---------------------------------------------------------------------------
PyObject* CPyCppyy::CPPFunction::Call(
        CPPInstance*& self, PyObject* args, PyObject* kwds, CallContext* ctxt)
{
// setup as necessary
    if (!fIsInitialized && !this->Initialize(ctxt))
        return nullptr;

// if function was attached to a class, self will be non-zero and should be
// the first function argument, so reorder 
    if (self) args = this->PreProcessArgs(self, args, kwds);

// translate the arguments as normal
    bool bConvertOk = this->ConvertAndSetArgs(args, ctxt);

    if (self) Py_DECREF(args);

    if (bConvertOk == false)
        return nullptr;

// execute function
    return this->Execute(nullptr, 0, ctxt);
}


//- CPPReverseBinary public members ---------------------------------------------
PyObject* CPyCppyy::CPPReverseBinary::PreProcessArgs(
        CPPInstance*& self, PyObject* args, PyObject* kwds)
{
    if (self) {
    // add self as part of the function arguments (means bound member)
        args = this->CPPFunction::PreProcessArgs(self, args, kwds);
    }

// swap the arguments
    PyObject* tmp = PyTuple_GET_ITEM(args, 0);
    PyTuple_SET_ITEM(args, 0, PyTuple_GET_ITEM(args, 1));
    PyTuple_SET_ITEM(args, 1, tmp);

    return args;
}

//---------------------------------------------------------------------------
PyObject* CPyCppyy::CPPReverseBinary::Call(
        CPPInstance*& self, PyObject* args, PyObject* kwds, CallContext* ctxt)
{
// This Call() function is very similar to the one of CPPFunction: only
// difference is that PreProcessArgs() is always called.

// setup as necessary
    if (!fIsInitialized && !this->Initialize(ctxt))
        return nullptr;

// if function was attached to a class, self will be non-zero and should be
// the first function argument, further, the arguments needs swapping
    args = this->PreProcessArgs(self, args, kwds);

// translate the arguments as normal
    bool bConvertOk = this->ConvertAndSetArgs(args, ctxt);

    if (self) Py_DECREF(args);

    if (bConvertOk == false)
        return nullptr;

// execute function
    return this->Execute(nullptr, 0, ctxt);
}

