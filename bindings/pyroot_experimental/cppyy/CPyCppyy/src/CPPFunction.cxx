// Bindings
#include "CPyCppyy.h"
#include "CPPFunction.h"
#include "CPPInstance.h"


//- public members --------------------------------------------------------------
PyObject* CPyCppyy::CPPFunction::PreProcessArgs(
        CPPInstance*& self, PyObject* args, PyObject*)
{
// no self means called as a free function; all ok
    if (!self) {
        Py_INCREF(args);
        return args;
    }

// otherwise, add self as part of the function arguments (means bound member)
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
// preliminary check in case keywords are accidently used (they are ignored otherwise)
    if (kwds && PyDict_Size(kwds)) {
        PyErr_SetString(PyExc_TypeError, "keyword arguments are not yet supported");
        return nullptr;
    }

// setup as necessary
    if (!this->Initialize(ctxt))
        return nullptr;

// reorder self into args, if necessary
    if (!(args = this->PreProcessArgs(self, args, kwds)))
        return nullptr;

// translate the arguments
    bool bConvertOk = this->ConvertAndSetArgs(args, ctxt);
    Py_DECREF(args);

    if (bConvertOk == false)
        return nullptr;

// execute function
    return this->Execute(nullptr, 0, ctxt);
}
