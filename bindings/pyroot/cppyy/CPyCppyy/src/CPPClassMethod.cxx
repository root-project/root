// Bindings
#include "CPyCppyy.h"
#include "CPPClassMethod.h"
#include "CPPInstance.h"


//- public members --------------------------------------------------------------
PyObject* CPyCppyy::CPPClassMethod::Call(CPPInstance*&
#if PY_VERSION_HEX >= 0x03080000
    self
#endif
    , CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt)
{
// preliminary check in case keywords are accidently used (they are ignored otherwise)
    if (kwds && ((PyDict_Check(kwds) && PyDict_Size(kwds)) ||
           (PyTuple_CheckExact(kwds) && PyTuple_GET_SIZE(kwds)))) {
        PyErr_SetString(PyExc_TypeError, "keyword arguments are not yet supported");
        return nullptr;
    }

// setup as necessary
    if (!this->Initialize(ctxt))
        return nullptr;

// translate the arguments
#if PY_VERSION_HEX >= 0x03080000
// TODO: The following is not robust and should be revisited e.g. by making CPPOverloads
// that have only CPPClassMethods be true Python classmethods? Note that the original
// implementation wasn't 100% correct either (e.g. static size() mapped to len()).
//
// As-is, if no suitable `self` is given (normal case), but the type of the first argument
// matches the enclosing scope of the class method and it isn't needed for the call, then
// assume that the method was (incorrectly) bound and so drop that instance from args.
    int nargs = (int)CPyCppyy_PyArgs_GET_SIZE(args, nargsf);
    if ((!self || (PyObject*)self == Py_None) && nargs) {
        PyObject* arg0 = CPyCppyy_PyArgs_GET_ITEM(args, 0);
        if ((CPPInstance_Check(arg0) && ((CPPInstance*)arg0)->ObjectIsA() == GetScope()) && \
                (fArgsRequired <= nargs-1)) {
            args   += 1;     // drops first argument
            nargsf -= 1;
        }
    }
#endif

    if (!this->ConvertAndSetArgs(args, nargsf, ctxt))
        return nullptr;

// execute function
    return this->Execute(nullptr, 0, ctxt);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPClassMethod::GetTypeName()
{
    PyObject* cppname = CPyCppyy_PyText_FromString((GetReturnTypeName() + " (*)").c_str());
    CPyCppyy_PyText_AppendAndDel(&cppname, GetSignature(false /* show_formalargs */));
    return cppname;
}
