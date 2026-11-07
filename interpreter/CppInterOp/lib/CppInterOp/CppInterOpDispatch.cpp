// Implementation of the dispatch symbol table. Maps function names to
// Cpp:: addresses for dlopen consumers via CppGetProcAddress.
// This is a library internal — it includes CppInterOp.h (not Dispatch.h)
// because it needs the Cpp:: function declarations.

#include "CppInterOp/CppInterOp.h"

#include <iostream>
#include <string_view>
#include <unordered_map>

using namespace Cpp;
using CppFnPtrTy = void (*)();

// NOLINTBEGIN(cppcoreguidelines-pro-type-cstyle-cast)
// Suppress deprecation: the dispatch table intentionally exposes all
// overloads, including those marked [[deprecated]] in the public API.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
static const std::unordered_map<std::string_view, CppFnPtrTy> DispatchMap = {
#define CPPINTEROP_API_FUNC(DN, CN, Ret, DeclArgs, CallArgs, RawTypes)         \
  {#DN, (CppFnPtrTy) static_cast<Ret(*) RawTypes>(&Cpp::CN)},
#include "CppInterOp/CppInterOpAPI.inc"
};
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
// NOLINTEND(cppcoreguidelines-pro-type-cstyle-cast)

extern "C" CPPINTEROP_API CppFnPtrTy CppGetProcAddress(const char* funcName) {
  auto it = DispatchMap.find(funcName);
  if (it == DispatchMap.end()) {
    std::cerr << "[CppInterOp Dispatch] Failed to find API: " << funcName
              << " May need to be ported to CppInterOp.td\n";
    return nullptr;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return reinterpret_cast<CppFnPtrTy>(it->second);
}
