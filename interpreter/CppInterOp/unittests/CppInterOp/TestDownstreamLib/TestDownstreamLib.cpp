#include "TestDownstreamLib.h"

#include "CppInterOp/Dispatch.h"

// Per-DSO slot storage, mirroring cppyy-backend's cppinterop_dispatch.cxx.
namespace CppInternal {
namespace DispatchRaw {
#define CPPINTEROP_API_FUNC(DN, CN, Ret, DeclArgs, CallArgs, RawTypes)         \
  Ret(*DN) RawTypes = nullptr;
#include "CppInterOp/CppInterOpAPI.inc"
} // namespace DispatchRaw
} // namespace CppInternal

// ODR-uses the inline JitCall fast path. JC is opaque so the optimizer
// can't DCE the calls at any -O level. The body never runs at test
// time; only the .o's UND-symbol surface matters.
void downstream_link_probe(CppImpl::JitCall* JC) {
  JC->Invoke();
  JC->InvokeConstructor(nullptr);
  JC->InvokeDestructor(nullptr);
}

int downstream_verify_trace_slots(const char* libpath) {
  if (!CppInternal::Dispatch::LoadDispatchAPI(libpath))
    return 1;
  if (!CppInternal::DispatchRaw::CppInterOpTraceJitCallInvokeImpl)
    return 2;
  if (!CppInternal::DispatchRaw::CppInterOpTraceJitCallInvokeDestructorImpl)
    return 3;
  if (!CppInternal::DispatchRaw::CppInterOpTraceJitCallInvokeReturnImpl)
    return 4;
  return 0;
}
