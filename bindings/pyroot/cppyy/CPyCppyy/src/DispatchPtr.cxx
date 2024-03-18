// Bindings
#include "CPyCppyy.h"
#define CPYCPPYY_INTERNAL 1
#include "CPyCppyy/DispatchPtr.h"
#undef CPYCPPYY_INTERNAL
#include "CPPInstance.h"
#include "CPPScope.h"


//-----------------------------------------------------------------------------
PyObject* CPyCppyy::DispatchPtr::Get() const
{
    if (fPyHardRef) return fPyHardRef;
    if (fPyWeakRef) {
        PyObject* disp = PyWeakref_GetObject(fPyWeakRef);
        if (disp != Py_None)      // dispatcher object disappeared?
            return disp;
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
    PyObject* pyobj = other.Get();
    fPyHardRef = pyobj ? (PyObject*)((CPPInstance*)pyobj)->Copy(cppinst) : nullptr;
    if (fPyHardRef) ((CPPInstance*)fPyHardRef)->SetDispatchPtr(this);
}

//-----------------------------------------------------------------------------
CPyCppyy::DispatchPtr::~DispatchPtr() {
// if we're holding a hard reference, or holding weak reference while being part
// of a dispatcher intermediate, then this delete is from the C++ side, and Python
// is "notified" by nulling out the reference and an exception will be raised on
// continued access
    if (fPyWeakRef) {
        PyObject* pyobj = PyWeakref_GetObject(fPyWeakRef);
        if (pyobj && pyobj != Py_None && ((CPPScope*)Py_TYPE(pyobj))->fFlags & CPPScope::kIsPython)
            ((CPPInstance*)pyobj)->GetObjectRaw() = nullptr;
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
        PyObject* pyobj = other.Get();
        fPyHardRef = pyobj ? (PyObject*)((CPPInstance*)pyobj)->Copy(cppinst) : nullptr;
        if (fPyHardRef) ((CPPInstance*)fPyHardRef)->SetDispatchPtr(this);
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
        fPyHardRef = PyWeakref_GetObject(fPyWeakRef);
        if (fPyHardRef == Py_None) fPyHardRef = nullptr;
        Py_XINCREF(fPyHardRef);
        Py_DECREF(fPyWeakRef); fPyWeakRef = nullptr;
    }
}
