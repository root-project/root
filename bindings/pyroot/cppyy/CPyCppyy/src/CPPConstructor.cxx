// Bindings
#include "CPyCppyy.h"
#include "CPPConstructor.h"
#include "CPPInstance.h"
#include "Executors.h"
#include "MemoryRegulator.h"
#include "ProxyWrappers.h"
#include "PyStrings.h"

// Standard
#include <memory>
#include <string>


//- data _____________________________________________________________________
namespace CPyCppyy {
    extern PyObject* gNullPtrObject;
}


//- protected members --------------------------------------------------------
bool CPyCppyy::CPPConstructor::InitExecutor_(Executor*& executor, CallContext*)
{
// pick up special case new object executor
    executor = CreateExecutor("__init__");
    return true;
}

//- public members -----------------------------------------------------------
PyObject* CPyCppyy::CPPConstructor::GetDocString()
{
// GetMethod() may return an empty function if this is just a special case place holder
    const std::string& clName = Cppyy::GetFinalName(this->GetScope());
    return CPyCppyy_PyText_FromFormat("%s::%s%s",
        clName.c_str(), clName.c_str(), this->GetMethod() ? this->GetSignatureString().c_str() : "()");
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPConstructor::Reflex(
    Cppyy::Reflex::RequestId_t request, Cppyy::Reflex::FormatId_t format)
{
// C++ reflection tooling for constructors.

    if (request == Cppyy::Reflex::RETURN_TYPE) {
        std::string fn = Cppyy::GetScopedFinalName(this->GetScope());
        if (format == Cppyy::Reflex::OPTIMAL || format == Cppyy::Reflex::AS_TYPE)
            return CreateScopeProxy(fn);
        else if (format == Cppyy::Reflex::AS_STRING)
            return CPyCppyy_PyText_FromString(fn.c_str());
    }

    return PyCallable::Reflex(request, format);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPConstructor::Call(CPPInstance*& self,
    CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt)
{

// setup as necessary
    if (fArgsRequired == -1 && !this->Initialize(ctxt))
        return nullptr;                     // important: 0, not Py_None

// fetch self, verify, and put the arguments in usable order
    PyCallArgs cargs{self, args, nargsf, kwds};
    if (!this->ProcessArgs(cargs))
        return nullptr;

// verify existence of self (i.e. tp_new called)
    if (!self) {
        PyErr_SetString(PyExc_ReferenceError, "no python object allocated");
        return nullptr;
    }

    if (self->GetObject()) {
        PyErr_SetString(PyExc_ReferenceError,
            "object already constructed; use __assign__ instead of __init__");
        return nullptr;
    }

    const auto cppScopeFlags = ((CPPScope*)Py_TYPE(self))->fFlags;

// Do nothing if the constructor is explicit and we are in an implicit
// conversion context. We recognize this by checking the CPPScope::kNoImplicit
// flag, as further implicit conversions are disabled to prevent infinite
// recursion. See also the ConvertImplicit() helper in Converters.cxx.
    if((cppScopeFlags & CPPScope::kNoImplicit) && Cppyy::IsExplicit(GetMethod()))
        return nullptr;

// self provides the python context for lifelines
    if (!ctxt->fPyContext)
        ctxt->fPyContext = (PyObject*)cargs.fSelf;    // no Py_INCREF as no ownership

// perform the call, nullptr 'this' makes the other side allocate the memory
    Cppyy::TCppScope_t disp = self->ObjectIsA(false /* check_smart */);
    intptr_t address = 0;
    if (GetScope() != disp) {
    // happens for Python derived types (which have a dispatcher inserted that
    // is not otherwise user-visible: call it instead) and C++ derived classes
    // without public constructors

    // first, check whether we at least had a proper meta class, or whether that
    // was also replaced user-side
        if (!GetScope() || !disp) {
            PyErr_SetString(PyExc_TypeError, "can not construct incomplete C++ class");
            return nullptr;
        }

    // get the dispatcher class and verify
        PyObject* dispproxy = CPyCppyy::GetScopeProxy(disp);
        if (!dispproxy) {
            PyErr_SetString(PyExc_TypeError, "dispatcher proxy was never created");
            return nullptr;
        }

        if (!(((CPPClass*)dispproxy)->fFlags & CPPScope::kIsPython)) {
            PyErr_SetString(PyExc_TypeError, const_cast<char*>((
                "constructor for " + Cppyy::GetScopedFinalName(disp) + " is not a dispatcher").c_str()));
            return nullptr;
        }

        PyObject* pyobj = CPyCppyy_PyObject_Call(dispproxy, cargs.fArgs, cargs.fNArgsf, kwds);
        if (!pyobj)
            return nullptr;

    // retrieve the actual pointer, take over control, and set set _internal_self
        address = (intptr_t)((CPPInstance*)pyobj)->GetObject();
        if (address) {
            ((CPPInstance*)pyobj)->CppOwns();    // b/c self will control the object on address
            PyObject* res = PyObject_CallMethodObjArgs(
                dispproxy, PyStrings::gDispInit, pyobj, (PyObject*)self, nullptr);
            Py_XDECREF(res);
        }
        Py_DECREF(dispproxy);

    } else {
    // translate the arguments
        if (cppScopeFlags & CPPScope::kNoImplicit)
            ctxt->fFlags |= CallContext::kNoImplicit;
        if (!this->ConvertAndSetArgs(cargs.fArgs, cargs.fNArgsf, ctxt))
            return nullptr;

        address = (intptr_t)this->Execute(nullptr, 0, ctxt);
    }

// return object if successful, lament if not
    if (address) {
        Py_INCREF(self);

    // note: constructors are no longer set to take ownership by default; instead that is
    // decided by the method proxy (which carries a creator flag) upon return
        self->Set((void*)address);

    // mark as actual to prevent needless auto-casting and register on its class
        self->fFlags |= CPPInstance::kIsActual;
        if (!(((CPPClass*)Py_TYPE(self))->fFlags & CPPScope::kIsSmart))
            MemoryRegulator::RegisterPyObject(self, (Cppyy::TCppObject_t)address);

    // handling smart types this way is deeply fugly, but if CPPInstance sets the proper
    // types in op_new first, then the wrong init is called
        if (((CPPClass*)Py_TYPE(self))->fFlags & CPPScope::kIsSmart) {
            PyObject* pyclass = CreateScopeProxy(((CPPSmartClass*)Py_TYPE(self))->fUnderlyingType);
            if (pyclass) {
                self->SetSmart((PyObject*)Py_TYPE(self));
                Py_DECREF((PyObject*)Py_TYPE(self));
                Py_SET_TYPE(self, (PyTypeObject*)pyclass);
            }
        }

    // done with self
        Py_DECREF(self);

        Py_RETURN_NONE;                     // by definition
    }

    if (!PyErr_Occurred())   // should be set, otherwise write a generic error msg
        PyErr_SetString(PyExc_TypeError, const_cast<char*>(
            (Cppyy::GetScopedFinalName(GetScope()) + " constructor failed").c_str()));

// do not throw an exception, nullptr might trigger the overload handler to
// choose a different constructor, which if all fails will throw an exception
    return nullptr;
}

//----------------------------------------------------------------------------
CPyCppyy::CPPMultiConstructor::CPPMultiConstructor(Cppyy::TCppScope_t scope, Cppyy::TCppMethod_t method) :
    CPPConstructor(scope, method)
{
    fNumBases = Cppyy::GetNumBases(scope);
}

//----------------------------------------------------------------------------
CPyCppyy::CPPMultiConstructor::CPPMultiConstructor(const CPPMultiConstructor& s) :
    CPPConstructor(s), fNumBases(s.fNumBases)
{
}

//----------------------------------------------------------------------------
CPyCppyy::CPPMultiConstructor& CPyCppyy::CPPMultiConstructor::operator=(const CPPMultiConstructor& s)
{
    if (this != &s) {
        CPPConstructor::operator=(s);
        fNumBases = s.fNumBases;
    }
    return *this;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPMultiConstructor::Call(CPPInstance*& self,
    CPyCppyy_PyArgs_t argsin, size_t nargsf, PyObject* kwds, CallContext* ctxt)
{
// By convention, initialization parameters of multiple base classes are grouped
// by target base class. Here, we disambiguate and put in "sentinel" parameters
// that allow the dispatcher to propagate them.

// Three options supported:
//  0. empty args: default constructor call
//  1. fNumBases tuples, each handed to individual constructors
//  2. less than fNumBases, assuming (void) for the missing base constructors
//  3. normal arguments, going to the first base only

// TODO: this way of forwarding is expensive as the loop is external to this call;
// it would be more efficient to have the argument handling happen beforehand

#if PY_VERSION_HEX >= 0x03080000
// fetch self, verify, and put the arguments in usable order (if self is not handled
// first, arguments can not be reordered with sentinels in place)
    PyCallArgs cargs{self, argsin, nargsf, kwds};
    if (!this->ProcessArgs(cargs))
        return nullptr;

// to re-use the argument handling, simply change the argument array into a tuple (the
// benefits of not allocating the tuple are relatively minor in this case)
    Py_ssize_t nargs = CPyCppyy_PyArgs_GET_SIZE(cargs.fArgs, cargs.fNArgsf);
    PyObject* args = PyTuple_New(nargs);
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        Py_INCREF(cargs.fArgs[i]);
        PyTuple_SET_ITEM(args, i, cargs.fArgs[i]);
    }

// copy out self as it may have been updated
    self = cargs.fSelf;

#else
    PyObject* args = argsin;
    Py_INCREF(args);
#endif

    if (PyTuple_CheckExact(args) && PyTuple_GET_SIZE(args)) {   // case 0. falls through
        Py_ssize_t nArgs = PyTuple_GET_SIZE(args);

        bool isAllTuples = true;
        Py_ssize_t nArgsTot = 0;
        for (Py_ssize_t i = 0; i < nArgs; ++i) {
            PyObject* argi = PyTuple_GET_ITEM(args, i);
            if (!PyTuple_CheckExact(argi)) {
                isAllTuples = false;
                break;
            }
            nArgsTot += PyTuple_GET_SIZE(argi);
        }

        if (isAllTuples) {
        // copy over the arguments, while filling in the sentinels (case 1. & 2.), with
        // just sentinels for the remaining (void) calls (case 2.)
            PyObject* newArgs = PyTuple_New(nArgsTot + fNumBases - 1);
            Py_ssize_t idx = 0;
            for (Py_ssize_t i = 0; i < nArgs; ++i) {
                if (i != 0) {
                // add sentinel
                    Py_INCREF(gNullPtrObject);
                    PyTuple_SET_ITEM(newArgs, idx, gNullPtrObject);
                    idx += 1;
                }

                PyObject* argi = PyTuple_GET_ITEM(args, i);
                for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(argi); ++j) {
                    PyObject* item = PyTuple_GET_ITEM(argi, j);
                    Py_INCREF(item);
                    PyTuple_SET_ITEM(newArgs, idx, item);
                    idx += 1;
                }
            }

        // add final sentinels as needed
            while (idx < (nArgsTot+fNumBases-1)) {
                Py_INCREF(gNullPtrObject);
                PyTuple_SET_ITEM(newArgs, idx, gNullPtrObject);
                idx += 1;
            }

            Py_DECREF(args);
            args = newArgs;
        } else {                                               // case 3. add sentinels
        // copy arguments as-is, then add sentinels at the end
            PyObject* newArgs = PyTuple_New(PyTuple_GET_SIZE(args) + fNumBases - 1);
            for (Py_ssize_t i = 0; i < nArgs; ++i) {
                PyObject* item = PyTuple_GET_ITEM(args, i);
                Py_INCREF(item);
                PyTuple_SET_ITEM(newArgs, i, item);
            }
            for (Py_ssize_t i = 0; i < fNumBases - 1; ++i) {
                Py_INCREF(gNullPtrObject);
                PyTuple_SET_ITEM(newArgs, i+nArgs, gNullPtrObject);
            }
            Py_DECREF(args);
            args = newArgs;
        }
    }

#if PY_VERSION_HEX < 0x03080000
    Py_ssize_t
#endif
    nargs = PyTuple_GET_SIZE(args);

#if PY_VERSION_HEX >= 0x03080000
// now unroll the new args tuple into a vector of objects
    auto argsu = std::unique_ptr<PyObject*[]>{new PyObject*[nargs]};
    for (Py_ssize_t i = 0; i < nargs; ++i)
        argsu[i] = PyTuple_GET_ITEM(args, i);
    CPyCppyy_PyArgs_t _args = argsu.get();
#else
    CPyCppyy_PyArgs_t _args = args;
#endif

    PyObject* result = CPPConstructor::Call(self, _args, nargs, kwds, ctxt);
    Py_DECREF(args);

    return result;
}


//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPAbstractClassConstructor::Call(CPPInstance*& self,
    CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt)
{
// do not allow instantiation of abstract classes
    if ((self && GetScope() != self->ObjectIsA()
#if PY_VERSION_HEX >= 0x03080000
        ) || (!self && !(ctxt->fFlags & CallContext::kFromDescr) && \
              CPyCppyy_PyArgs_GET_SIZE(args, nargsf) && CPPInstance_Check(args[0]) && \
              GetScope() != ((CPPInstance*)args[0])->ObjectIsA()
#endif
        )) {
    // happens if a dispatcher is inserted; allow constructor call
        return CPPConstructor::Call(self, args, nargsf, kwds, ctxt);
    }

    PyErr_Format(PyExc_TypeError, "cannot instantiate abstract class \'%s\'"
            " (from derived classes, use super() instead)",
        Cppyy::GetScopedFinalName(this->GetScope()).c_str());
    return nullptr;
}


//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPNamespaceConstructor::Call(
    CPPInstance*&, CPyCppyy_PyArgs_t, size_t, PyObject*, CallContext*)
{
// do not allow instantiation of namespaces
    PyErr_Format(PyExc_TypeError, "cannot instantiate namespace \'%s\'",
        Cppyy::GetScopedFinalName(this->GetScope()).c_str());
    return nullptr;
}


//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPIncompleteClassConstructor::Call(
    CPPInstance*&, CPyCppyy_PyArgs_t, size_t, PyObject*, CallContext*)
{
// do not allow instantiation of incomplete (forward declared) classes)
    PyErr_Format(PyExc_TypeError, "cannot instantiate incomplete class \'%s\'",
        Cppyy::GetScopedFinalName(this->GetScope()).c_str());
    return nullptr;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPAllPrivateClassConstructor::Call(
    CPPInstance*&, CPyCppyy_PyArgs_t, size_t, PyObject*, CallContext*)
{
// do not allow instantiation of classes with only private constructors
    PyErr_Format(PyExc_TypeError, "cannot instantiate class \'%s\' that has no public constructors",
        Cppyy::GetScopedFinalName(this->GetScope()).c_str());
    return nullptr;
}
