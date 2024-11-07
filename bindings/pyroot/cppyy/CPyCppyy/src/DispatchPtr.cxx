// Bindings
#include "CPyCppyy.h"
#define CPYCPPYY_INTERNAL 1
#include "CPyCppyy/DispatchPtr.h"
#undef CPYCPPYY_INTERNAL
#include "CPPInstance.h"
#include "CPPScope.h"


//-----------------------------------------------------------------------------
PyObject* CPyCppyy::DispatchPtr::Get(bool borrowed) const
{
    if (fPyHardRef) {
        if (!borrowed) Py_INCREF(fPyHardRef);
        return fPyHardRef;
    }
    if (fPyWeakRef) {
        PyObject* disp = CPyCppyy_GetWeakRef(fPyWeakRef);
        if (disp) {               // dispatcher object disappeared?
            if (borrowed) Py_DECREF(disp);
            return disp;
        }
    }
    return nullptr;
}

//-----------------------------------------------------------------------------
CPyCppyy::DispatchPtr::DispatchPtr(PyObject* pyobj, bool strong) : fPyHardRef(nullptr)
{
    if (strong) {
        Py_INCREF(pyobj);
        fPyHardRef = pyobj;
        fPyWeakRef = nullptr;
    } else {
        fPyHardRef = nullptr;
        fPyWeakRef = PyWeakref_NewRef(pyobj, nullptr);
    }
    ((CPPInstance*)pyobj)->SetDispatchPtr(this);
}

//-----------------------------------------------------------------------------
CPyCppyy::DispatchPtr::DispatchPtr(const DispatchPtr& other, void* cppinst) : fPyWeakRef(nullptr)
{
    PyObject* pyobj = other.Get(false /* not borrowed */);
    fPyHardRef = pyobj ? (PyObject*)((CPPInstance*)pyobj)->Copy(cppinst) : nullptr;
    if (fPyHardRef) ((CPPInstance*)fPyHardRef)->SetDispatchPtr(this);
    Py_XDECREF(pyobj);
}

//-----------------------------------------------------------------------------
CPyCppyy::DispatchPtr::~DispatchPtr() {
// if we're holding a hard reference, or holding weak reference while being part
// of a dispatcher intermediate, then this delete is from the C++ side, and Python
// is "notified" by nulling out the reference and an exception will be raised on
// continued access
    if (fPyWeakRef) {
        PyObject* pyobj = CPyCppyy_GetWeakRef(fPyWeakRef);
        if (pyobj && ((CPPScope*)Py_TYPE(pyobj))->fFlags & CPPScope::kIsPython)
            ((CPPInstance*)pyobj)->GetObjectRaw() = nullptr;
        Py_XDECREF(pyobj);
        Py_DECREF(fPyWeakRef);
    } else if (fPyHardRef) {
        ((CPPInstance*)fPyHardRef)->GetObjectRaw() = nullptr;
        Py_DECREF(fPyHardRef);
    }
}

//-----------------------------------------------------------------------------
CPyCppyy::DispatchPtr& CPyCppyy::DispatchPtr::assign(const DispatchPtr& other, void* cppinst)
{
    if (this != &other) {
        Py_XDECREF(fPyWeakRef); fPyWeakRef = nullptr;
        Py_XDECREF(fPyHardRef);
        PyObject* pyobj = other.Get(false /* not borrowed */);
        fPyHardRef = pyobj ? (PyObject*)((CPPInstance*)pyobj)->Copy(cppinst) : nullptr;
        if (fPyHardRef) ((CPPInstance*)fPyHardRef)->SetDispatchPtr(this);
        Py_XDECREF(pyobj);
    }
    return *this;
}

//-----------------------------------------------------------------------------
void CPyCppyy::DispatchPtr::PythonOwns()
{
// Python maintains the hardref, so only allowed a weakref here
    if (fPyHardRef) {
        fPyWeakRef = PyWeakref_NewRef(fPyHardRef, nullptr);
        Py_DECREF(fPyHardRef); fPyHardRef = nullptr;
    }
}

//-----------------------------------------------------------------------------
void CPyCppyy::DispatchPtr::CppOwns()
{
// C++ maintains the hardref, keeping the PyObject alive w/o outstanding ref
    if (fPyWeakRef) {
        fPyHardRef = CPyCppyy_GetWeakRef(fPyWeakRef);
        Py_DECREF(fPyWeakRef); fPyWeakRef = nullptr;
    }
}
