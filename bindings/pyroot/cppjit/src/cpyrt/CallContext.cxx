// Bindings
#include "cpyrt.h"

using namespace cppjit;
#include "CallContext.h"

//-----------------------------------------------------------------------------
uint32_t& cpyrt::CallContext::GlobalPolicyFlags() {
  static uint32_t flags = 0;
  return flags;
}

//-----------------------------------------------------------------------------
void cpyrt::CallContext::AddTemporary(PyObject* pyobj) {
  if (pyobj) {
    if (!fTemps)
      fTemps = new Temporary{pyobj, nullptr};
    else {
      Temporary* tmp = fTemps;
      while (tmp->fNext)
        tmp = tmp->fNext;
      tmp->fNext = new Temporary{pyobj, nullptr};
    }
  }
}

//-----------------------------------------------------------------------------
void cpyrt::CallContext::Cleanup() {
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
bool cpyrt::CallContext::SetGlobalPolicy(ECallFlags toggleFlag, bool enabled) {
  auto& flags = GlobalPolicyFlags();
  bool old = flags & toggleFlag;
  if (enabled)
    flags |= toggleFlag;
  else
    flags &= ~toggleFlag;
  return old;
}
