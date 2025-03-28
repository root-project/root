// Bindings
#include "CPyCppyy.h"
#include "CPPOperator.h"
#include "CPPInstance.h"
#include "Utility.h"


//- constructor --------------------------------------------------------------
CPyCppyy::CPPOperator::CPPOperator(
    Cppyy::TCppScope_t scope, Cppyy::TCppMethod_t method, const std::string& name)
        : CPPMethod(scope, method)
{
// a bit silly but doing it this way allows decoupling the initialization order
    if (name == "__mul__")
        fStub = CPPInstance_Type.tp_as_number->nb_multiply;
    else if (name == CPPYY__div__)
#if PY_VERSION_HEX < 0x03000000
        fStub = CPPInstance_Type.tp_as_number->nb_divide;
#else
        fStub = CPPInstance_Type.tp_as_number->nb_true_divide;
#endif
    else if (name == "__add__")
        fStub = CPPInstance_Type.tp_as_number->nb_add;
    else if (name == "__sub__")
        fStub = CPPInstance_Type.tp_as_number->nb_subtract;
    else
        fStub = nullptr;
}

//-----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPOperator::Call(CPPInstance*& self,
    CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt)
{
// some operators can be a mix of global and class overloads; this method will
// first try class overloads (the existence of this method means that such were
// defined) and if failed, fall back on the global stubs
// TODO: the fact that this is a method and not an overload means that the global
// ones are tried for each method that fails during the overload resolution
    PyObject* result = this->CPPMethod::Call(self, args, nargsf, kwds, ctxt);
    if (result || !fStub || !self)
        return result;

    Py_ssize_t idx_other = 0;
    if (CPyCppyy_PyArgs_GET_SIZE(args, nargsf) != 1) {
#if PY_VERSION_HEX >= 0x03080000
        if ((CPyCppyy_PyArgs_GET_SIZE(args, nargsf) == 2 && CPyCppyy_PyArgs_GET_ITEM(args, 0) == (PyObject*)self))
            idx_other = 1;
        else
#endif
        return result;
    }

// fetch the current error, resetting the error buffer
    auto error = CPyCppyy::Utility::FetchPyError();

    result = fStub((PyObject*)self, CPyCppyy_PyArgs_GET_ITEM(args, idx_other));

// if there was still a problem, restore the Python error buffer
    if (!result) {
        CPyCppyy::Utility::RestorePyError(error);
    }

    return result;
}
