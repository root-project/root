// Bindings
#include "CPyCppyy.h"
#include "CPPGetSetItem.h"
#include "Executors.h"


//- private helpers -----------------------------------------------------------
static inline
void unroll(CPyCppyy_PyArgs_t packed_args, CPyCppyy_PyArgs_t unrolled, Py_ssize_t nArgs)
{
// Unroll up to nArgs arguments from packed_args into unrolled.
    for (int i = 0, iur = 0; i < nArgs; ++i, ++iur) {
        PyObject* item = CPyCppyy_PyArgs_GET_ITEM(packed_args, i);
        if (PyTuple_Check(item)) {
            for (int j = 0; j < PyTuple_GET_SIZE(item); ++j, ++iur) {
                PyObject* subitem = PyTuple_GET_ITEM(item, j);
                Py_INCREF(subitem);
                CPyCppyy_PyArgs_SET_ITEM(unrolled, iur, subitem);
            }
        } else {
            Py_INCREF(item);
            CPyCppyy_PyArgs_SET_ITEM(unrolled, iur, item);
        }
    }
}

//- protected members ---------------------------------------------------------
bool CPyCppyy::CPPSetItem::InitExecutor_(Executor*& executor, CallContext*)
{
// basic call will do
    if (!CPPMethod::InitExecutor_(executor))
        return false;

// check to make sure we're dealing with a RefExecutor
    if (!dynamic_cast<RefExecutor*>(executor)) {
        PyErr_Format(PyExc_NotImplementedError,
            "no __setitem__ handler for return type (%s)",
            this->GetReturnTypeName().c_str());
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool CPyCppyy::CPPSetItem::ProcessArgs(PyCallArgs& cargs)
{
// Prepare executor with a buffer for the return value.
    Py_ssize_t nArgs = CPyCppyy_PyArgs_GET_SIZE(cargs.fArgs, cargs.fNArgsf);
    if (nArgs <= 1) {
        PyErr_SetString(PyExc_TypeError, "insufficient arguments to __setitem__");
        return false;
    }

// use the last element of args for assignment upon return, then slice it from
// the (unrolled) actual arguments
    ((RefExecutor*)this->GetExecutor())->SetAssignable(CPyCppyy_PyArgs_GET_ITEM(cargs.fArgs, nArgs-1));

// see whether any of the arguments is a tuple to be unrolled
    Py_ssize_t realsize = 0;
    for (Py_ssize_t i = 0; i < nArgs-1; ++i) {
        PyObject* item = CPyCppyy_PyArgs_GET_ITEM(cargs.fArgs, i);
        realsize += PyTuple_Check(item) ? PyTuple_GET_SIZE(item) : 1;
    }

// unroll any tuples, if present in the arguments
#if PY_VERSION_HEX >= 0x03080000
    if (realsize != nArgs-1) {
        CPyCppyy_PyArgs_t unrolled = (PyObject**)PyMem_Malloc(realsize * sizeof(PyObject*));
        unroll(cargs.fArgs, unrolled, nArgs-1);
        cargs.fArgs = unrolled;
        cargs.fFlags |= PyCallArgs::kDoFree;
    }
#else
    if (realsize != nArgs-1) {
        CPyCppyy_PyArgs_t unrolled = PyTuple_New(realsize);
        unroll(cargs.fArgs, unrolled, nArgs-1);
        cargs.fArgs = unrolled;
    } else
        cargs.fArgs = PyTuple_GetSlice(cargs.fArgs, 0, nArgs-1);
    cargs.fFlags |= PyCallArgs::kDoDecref;
#endif
    cargs.fNArgsf = realsize;

// continue normal method processing
    return CPPMethod::ProcessArgs(cargs);
}


//-----------------------------------------------------------------------------
bool CPyCppyy::CPPGetItem::ProcessArgs(PyCallArgs& cargs)
{
// Unroll tuples for call, otherwise just like regular CPPMethod of __getitem__.
    Py_ssize_t nArgs = CPyCppyy_PyArgs_GET_SIZE(cargs.fArgs, cargs.fNArgsf);

// see whether any of the arguments is a tuple to be unrolled
    Py_ssize_t realsize = 0;
    for (Py_ssize_t i = 0; i < nArgs; ++i) {
        PyObject* item = CPyCppyy_PyArgs_GET_ITEM(cargs.fArgs, i);
        realsize += PyTuple_Check(item) ? PyTuple_GET_SIZE(item) : 1;
    }

// unroll any tuples, if present in the arguments
    if (realsize != nArgs) {
        CPyCppyy_PyArgs_t packed_args = cargs.fArgs;
#if PY_VERSION_HEX >= 0x03080000
        cargs.fArgs = (PyObject**)PyMem_Malloc(realsize * sizeof(PyObject*));
        cargs.fFlags |= PyCallArgs::kDoFree;
#else
        cargs.fArgs = PyTuple_New(realsize);
        cargs.fFlags |= PyCallArgs::kDoDecref;
#endif
        cargs.fNArgsf = realsize;
        unroll(packed_args, cargs.fArgs, nArgs);
    }

// continue normal method processing
    return CPPMethod::ProcessArgs(cargs);
}
