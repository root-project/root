// Bindings
#include "CPyCppyy.h"
#include "CallContext.h"

namespace {

bool setPolicy(CPyCppyy::CallContext::ECallFlags toggleFlag, bool enabled) {
    auto &flags = CPyCppyy::CallContext::globalPolicyFlags();
    bool old = flags & toggleFlag;
    if (enabled)
        flags |= toggleFlag;
    else
        flags &= ~toggleFlag;
    return old;
}

} // namespace 


uint32_t &CPyCppyy::CallContext::globalPolicyFlags() {
   static uint32_t flags = 0;
   return flags;
}

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
bool CPyCppyy::CallContext::SetHeuristicMemoryPolicy(bool enabled)
{
// Set the global memory policy, which affects object ownership when objects
// are passed as function arguments.
    return setPolicy(kUseHeuristics, enabled);
}

//-----------------------------------------------------------------------------
bool CPyCppyy::CallContext::SetGlobalSignalPolicy(bool enabled)
{
// Set the global signal policy, which determines whether a jmp address
// should be saved to return to after a C++ segfault.
    return setPolicy(kProtected, enabled);
}

