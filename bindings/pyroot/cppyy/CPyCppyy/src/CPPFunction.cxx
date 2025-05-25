// Bindings
#include "CPyCppyy.h"
#include "CPPFunction.h"
#include "CPPInstance.h"

// Standard
#include <algorithm>


//- CFunction helpers -----------------------------------------------------------
bool CPyCppyy::AdjustSelf(PyCallArgs& cargs)
{
#if PY_VERSION_HEX >= 0x03080000
    if (cargs.fNArgsf & PY_VECTORCALL_ARGUMENTS_OFFSET) {  // mutation allowed?
        std::swap(((PyObject**)cargs.fArgs-1)[0], (PyObject*&)cargs.fSelf);
        cargs.fFlags |= PyCallArgs::kSelfSwap;
        cargs.fArgs -= 1;
        cargs.fNArgsf &= ~PY_VECTORCALL_ARGUMENTS_OFFSET;
        cargs.fNArgsf += 1;
    } else {
        Py_ssize_t nkwargs = cargs.fKwds ? PyTuple_GET_SIZE(cargs.fKwds) : 0;
        Py_ssize_t totalargs = PyVectorcall_NARGS(cargs.fNArgsf)+nkwargs;
        PyObject** newArgs = (PyObject**)PyMem_Malloc((totalargs+1) * sizeof(PyObject*));
        if (!newArgs)
            return false;

        newArgs[0] = (PyObject*)cargs.fSelf;
        if (0 < totalargs)
            memcpy((void*)&newArgs[1], cargs.fArgs, totalargs * sizeof(PyObject*));
        cargs.fArgs = newArgs;
        cargs.fFlags |= PyCallArgs::kDoFree;
        cargs.fNArgsf += 1;
    }
#else
    Py_ssize_t sz = PyTuple_GET_SIZE(cargs.fArgs);
    CPyCppyy_PyArgs_t newArgs = PyTuple_New(sz+1);
    for (int i = 0; i < sz; ++i) {
        PyObject* item = PyTuple_GET_ITEM(cargs.fArgs, i);
        Py_INCREF(item);
        PyTuple_SET_ITEM(newArgs, i+1, item);
    }
    Py_INCREF(cargs.fSelf);
    PyTuple_SET_ITEM(newArgs, 0, (PyObject*)cargs.fSelf);

    cargs.fArgs = newArgs;
    cargs.fFlags |= PyCallArgs::kDoDecref;
    cargs.fNArgsf += 1;
#endif
    return true;
}

bool CPyCppyy::CPPFunction::ProcessArgs(PyCallArgs& cargs)
{
// add self as part of the function arguments (means bound member)
    if (cargs.fKwds)
        return this->ProcessKwds((PyObject*)cargs.fSelf, cargs);
    return AdjustSelf(cargs);
}

//- CPPFunction public members --------------------------------------------------
PyObject* CPyCppyy::CPPFunction::Call(CPPInstance*& self,
    CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt)
{
// setup as necessary
    if (fArgsRequired == -1 && !this->Initialize(ctxt))
        return nullptr;

// if function was attached to a class, self will be non-zero and should be
// the first function argument, so reorder
    PyCallArgs cargs{self, args, nargsf, kwds};
    if (self || kwds) {
        if (!this->ProcessArgs(cargs))
            return nullptr;
    }

#if PY_VERSION_HEX >= 0x03080000
// special case, if this method was inserted as a constructor, then self is nullptr
// and it will be the first argument and needs to be used as Python context
    if (IsConstructor(ctxt->fFlags) && !ctxt->fPyContext && \
            CPyCppyy_PyArgs_GET_SIZE(cargs.fArgs, cargs.fNArgsf)) {
        ctxt->fPyContext = cargs.fArgs[0];
    }
#endif

// translate the arguments as normal
    if (!this->ConvertAndSetArgs(cargs.fArgs, cargs.fNArgsf, ctxt))
        return nullptr;

// execute function
    PyObject* result = this->Execute(nullptr, 0, ctxt);

#if PY_VERSION_HEX >= 0x03080000
// special case, if this method was inserted as a constructor, then if no self was
// provided, it will be the first argument and may have been updated
    if (IsConstructor(ctxt->fFlags) && result && !cargs.fSelf && \
            CPyCppyy_PyArgs_GET_SIZE(cargs.fArgs, cargs.fNArgsf) && CPPInstance_Check(cargs.fArgs[0])) {
        self = (CPPInstance*)cargs.fArgs[0];
        Py_INCREF(self);
    }
#endif

    return result;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPFunction::GetTypeName()
{
    PyObject* cppname = CPyCppyy_PyText_FromString((GetReturnTypeName() + " (*)").c_str());
    CPyCppyy_PyText_AppendAndDel(&cppname, GetSignature(false /* show_formalargs */));
    return cppname;
}


//- CPPReverseBinary private helper ---------------------------------------------
bool CPyCppyy::CPPReverseBinary::ProcessArgs(PyCallArgs& cargs)
{
    if (cargs.fSelf || cargs.fKwds) {
    // add self as part of the function arguments (means bound member)
        if (!this->CPPFunction::ProcessArgs(cargs))
            return false;
    }

// swap the arguments
#if PY_VERSION_HEX >= 0x03080000
    std::swap(((PyObject**)cargs.fArgs)[0], ((PyObject**)cargs.fArgs)[1]);
#else
    std::swap(PyTuple_GET_ITEM(cargs.fArgs, 0), PyTuple_GET_ITEM(cargs.fArgs, 1));
#endif
    cargs.fFlags |= PyCallArgs::kArgsSwap;

    return true;
}

//- CPPReverseBinary public members ---------------------------------------------
PyObject* CPyCppyy::CPPReverseBinary::Call(CPPInstance*& self,
    CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt)
{
// This Call() function is very similar to the one of CPPFunction: only difference is
// that ProcessArgs() is always called.

// setup as necessary
    if (fArgsRequired == -1 && !this->Initialize(ctxt))
        return nullptr;

// if function was attached to a class, self will be non-zero and should be
// the first function argument, further, the arguments needs swapping
    PyCallArgs cargs{self, args, nargsf, kwds};
    if (!this->ProcessArgs(cargs))
        return nullptr;

// translate the arguments as normal
    if (!this->ConvertAndSetArgs(cargs.fArgs, cargs.fNArgsf, ctxt))
        return nullptr;

// execute function
    return this->Execute(nullptr, 0, ctxt);
}
