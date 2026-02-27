// Bindings
#include "CPyCppyy.h"
#include "CallContext.h"

//-----------------------------------------------------------------------------
uint32_t &CPyCppyy::CallContext::GlobalPolicyFlags()
{
   static uint32_t flags = 0;
   return flags;
}

//- data _____________________________________________________________________
namespace CPyCppyy {

    CallContext::ECallFlags CallContext::sMemoryPolicy = CallContext::kUseStrict;
// this is just a data holder for linking; actual value is set in CPyCppyyModule.cxx
    CallContext::ECallFlags CallContext::sSignalPolicy = CallContext::kNone;

} // namespace CPyCppyy

//-----------------------------------------------------------------------------
void CPyCppyy::CallContext::AddTemporary(PyObject* pyobj) {
    if (pyobj) {
        if (!fTemps)
            fTemps = new Temporary{pyobj, nullptr};
        else {
            Temporary* tmp = fTemps;
            while (tmp->fNext) tmp = tmp->fNext;
            tmp->fNext = new Temporary{pyobj, nullptr};
        }
    }
}

//-----------------------------------------------------------------------------
void CPyCppyy::CallContext::Cleanup() {
    Temporary* tmp = fTemps;
    while (tmp) {
        Py_DECREF(tmp->fPyObject);
        Temporary* tmp2 = tmp->fNext;
        delete tmp;
        tmp = tmp2;
    }
    fTemps = nullptr;
}

//-----------------------------------------------------------------------------
bool CPyCppyy::CallContext::SetMemoryPolicy(ECallFlags e)
{
// Set the global memory policy, which affects object ownership when objects
// are passed as function arguments.
    if (kUseHeuristics == e || e == kUseStrict) {
        sMemoryPolicy = e;
        return true;
    }
    return false;
}

//-----------------------------------------------------------------------------
bool CPyCppyy::CallContext::SetGlobalSignalPolicy(bool setProtected)
{
// Set the global signal policy, which determines whether a jmp address
// should be saved to return to after a C++ segfault.
    bool old = sSignalPolicy == kProtected;
    sSignalPolicy = setProtected ? kProtected : kNone;
    return old;
}
bool CPyCppyy::CallContext::SetGlobalPolicy(ECallFlags toggleFlag, bool enabled)
{
    auto &flags = GlobalPolicyFlags();
    bool old = flags & toggleFlag;
    if (enabled)
        flags |= toggleFlag;
    else
        flags &= ~toggleFlag;
    return old;
}
