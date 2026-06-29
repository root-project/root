// Hand-written C API wrappers for functions the TableGen emitter
// cannot generate mechanically, plus C-ABI-shaped overloads of Cpp::
// APIs whose primary form (in CppInterOp.cpp) cannot cross the C
// boundary. Public declarations of these symbols live in
// include/CppInterOp/CXCppInterOp.h; generated wrappers live in
// CXCppInterOpGenerated.cpp.

#include "CppInterOp/CXCppInterOp.h"
#include "Unwrap.h"
#include "CppInterOp/CppInterOp.h"

#include "Compatibility.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#define CPPINTEROP_MSAN_UNPOISON_VALUE(v) __msan_unpoison(&(v), sizeof(v))
#else
#define CPPINTEROP_MSAN_UNPOISON_VALUE(v) ((void)0)
#endif
#else
#define CPPINTEROP_MSAN_UNPOISON_VALUE(v) ((void)0)
#endif

namespace Cpp {

// Legacy C-ABI overload of Cpp::Evaluate. The Box-returning overload in
// CppInterOp.cpp cannot cross the C boundary; bindings that go through
// the generated cppinterop_Evaluate_intptr wrapper (e.g. cppyy) land
// here instead.
intptr_t Evaluate(const char* code, bool* HadError) {
  auto* I = unwrap<compat::Interpreter>(GetInterpreter());
  compat::Value V;

  if (HadError)
    *HadError = false;

  auto res = I->evaluate(code, V);
  CPPINTEROP_MSAN_UNPOISON_VALUE(V);
  if (res != 0 || !V.hasValue()) {
    if (HadError)
      *HadError = true;
    // FIXME: Make this return llvm::Expected.
    return ~0UL;
  }

  return compat::convertTo<intptr_t>(V);
}

} // namespace Cpp

extern "C" {

// C-ABI bridge for the intptr_t Cpp::Evaluate overload above. The Box-
// returning overload generated from CppInterOp.td is C++-only.
CPPINTEROP_API intptr_t cppinterop_Evaluate(const char* code, bool* HadError) {
  return Cpp::Evaluate(code, HadError);
}

// GetClassTemplatedMethods returns bool AND fills a vector out-param.
// The C wrapper drops the bool (caller checks arr.size > 0 instead).
CPPINTEROP_API Cpp::CppInterOpArray
cppinterop_GetClassTemplatedMethods(const char* name, CppConstDeclRef parent) {
  std::vector<Cpp::FuncRef> out;
  Cpp::GetClassTemplatedMethods(std::string(name), parent, out);
  Cpp::CppInterOpArray arr = {nullptr, out.size()};
  if (arr.size) {
    arr.data = static_cast<void**>(malloc(arr.size * sizeof(void*)));
    memcpy(arr.data, out.data(), arr.size * sizeof(void*));
  }
  return arr;
}

} // extern "C"
