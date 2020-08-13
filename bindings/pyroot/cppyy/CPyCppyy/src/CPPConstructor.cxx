// Bindings
#include "CPyCppyy.h"
#include "CPPConstructor.h"
#include "CPPInstance.h"
#include "Executors.h"
#include "MemoryRegulator.h"
#include "ProxyWrappers.h"

#include "CPyCppyy/DispatchPtr.h"

// Standard
#include <string>


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
PyObject* CPyCppyy::CPPConstructor::Call(
    CPPInstance*& self, PyObject* args, PyObject* kwds, CallContext* ctxt)
{
// setup as necessary
    if (fArgsRequired == -1 && !this->Initialize(ctxt))
        return nullptr;                     // important: 0, not Py_None

// fetch self, verify, and put the arguments in usable order
    if (!(args = this->PreProcessArgs(self, args, kwds)))
        return nullptr;

    if (self->GetObject()) {
        Py_DECREF(args);
        PyErr_SetString(PyExc_ReferenceError,
            "object already constructed; use __assign__ instead of __init__");
        return nullptr;
    }

// translate the arguments
    if (!this->ConvertAndSetArgs(args, ctxt)) {
        Py_DECREF(args);
        return nullptr;
    }

// verify existence of self (i.e. tp_new called)
    if (!self) {
        PyErr_Print();
        PyErr_SetString(PyExc_ReferenceError, "no python object allocated");
        return nullptr;
    }

// perform the call, nullptr 'this' makes the other side allocate the memory
    Cppyy::TCppScope_t disp = self->ObjectIsA(false /* check_smart */);
    Cppyy::TCppMethod_t curMethod = GetMethod();
    if (GetScope() != disp) {
    // happens for Python derived types, which have a dispatcher inserted that
    // is not otherwise user-visible: temporarily reset fMethod
    // TODO: these lookups are slow and need caching
        const std::string& dispName = Cppyy::GetFinalName(disp);
    // select proper overload based on signature match
        const std::string& sig =
            (curMethod && PyTuple_GET_SIZE(args)) ? Cppyy::GetMethodSignature(curMethod, false, PyTuple_GET_SIZE(args)) : "()";
        Cppyy::TCppMethod_t method = (Cppyy::TCppMethod_t)0;
        const auto& v = Cppyy::GetMethodIndicesFromName(disp, dispName);
        for (auto idx : v) {
            Cppyy::TCppMethod_t mm = Cppyy::GetMethod(disp, idx);
            if (Cppyy::GetMethodSignature(mm, false) == sig) {
                method = mm;
                break;
            }
        }

        if (method) SetMethod(method);
        else {
            PyErr_Format(PyExc_TypeError, "no constructor available for \'%s\'",
                Cppyy::GetScopedFinalName(this->GetScope()).c_str());
            return nullptr;
        }
    }
    ptrdiff_t address = (ptrdiff_t)this->Execute(nullptr, 0, ctxt);
    if (GetMethod() != curMethod) {
    // restore the original constructor
        SetMethod(curMethod);

    // set m_self (TODO: get this from the compiler in case of some unorthodox padding
    // or if the inheritance hierarchy extends back into C++ land)
        if (address) {
            ptrdiff_t self_address = address + Cppyy::SizeOf(GetScope());
            new ((void*)self_address) DispatchPtr{(PyObject*)self};
        }
    }

// done with filtered args
    Py_DECREF(args);

// return object if successful, lament if not
    if (address) {
        Py_INCREF(self);

    // note: constructors are no longer set to take ownership by default; instead that is
    // decided by the method proxy (which carries a creator flag) upon return
        self->Set((void*)address);

    // TODO: consistent up or down cast ...
        MemoryRegulator::RegisterPyObject(self, (Cppyy::TCppObject_t)address);

    // handling smart types this way is deeply fugly, but if CPPInstance sets the proper
    // types in op_new first, then the wrong init is called
        if (((CPPClass*)Py_TYPE(self))->fFlags & CPPScope::kIsSmart) {
            PyObject* pyclass = CreateScopeProxy(((CPPSmartClass*)Py_TYPE(self))->fUnderlyingType);
            if (pyclass) {
                self->SetSmart((PyObject*)Py_TYPE(self));
                Py_DECREF((PyObject*)Py_TYPE(self));
                Py_TYPE(self) = (PyTypeObject*)pyclass;
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
PyObject* CPyCppyy::CPPAbstractClassConstructor::Call(
    CPPInstance*& self, PyObject* args, PyObject* kwds, CallContext* ctxt)
{
// do not allow instantiation of abstract classes
    if (self && GetScope() != self->ObjectIsA()) {
    // happens if a dispatcher is inserted; allow constructor call
        return CPPConstructor::Call(self, args, kwds, ctxt);
    }

    PyErr_Format(PyExc_TypeError, "cannot instantiate abstract class \'%s\'"
            " (from derived classes, use super() instead)",
        Cppyy::GetScopedFinalName(this->GetScope()).c_str());
    return nullptr;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPNamespaceConstructor::Call(
    CPPInstance*&, PyObject*, PyObject*, CallContext*)
{
// do not allow instantiation of namespaces
    PyErr_Format(PyExc_TypeError, "cannot instantiate namespace \'%s\'",
        Cppyy::GetScopedFinalName(this->GetScope()).c_str());
    return nullptr;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPIncompleteClassConstructor::Call(
    CPPInstance*&, PyObject*, PyObject*, CallContext*)
{
// do not allow instantiation of incomplete (forward declared) classes)
    PyErr_Format(PyExc_TypeError, "cannot instantiate incomplete class \'%s\'",
        Cppyy::GetScopedFinalName(this->GetScope()).c_str());
    return nullptr;
}
