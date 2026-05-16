// Storage definitions and initialization for dispatch function pointers.
// Linked only by DispatchTests — not by the normal test suite.

#include "CppInterOp/Dispatch.h"

// Define storage for all raw dispatch function pointers.
using namespace CppImpl;
#define CPPINTEROP_API_FUNC(DN, CN, Ret, DeclArgs, CallArgs, RawTypes)         \
  Ret(*CppInternal::DispatchRaw::DN) RawTypes = nullptr;
#include "CppInterOp/CppInterOpAPI.inc"

namespace {
struct DispatchInitializer {
  DispatchInitializer() {
    if (!Cpp::LoadDispatchAPI(CPPINTEROP_LIB_PATH)) {
      std::abort();
    }
  }
};
DispatchInitializer& GetDispatchInitializer() {
  static DispatchInitializer instance;
  return instance;
}
const DispatchInitializer& g_dispatch_init = GetDispatchInitializer();
} // namespace
