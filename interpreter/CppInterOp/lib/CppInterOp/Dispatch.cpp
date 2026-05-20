#include <CppInterOp/Dispatch.h>

#include <iostream> // for std::cerr
#include <string_view>
#include <unordered_map>

// NOLINTBEGIN(cppcoreguidelines-pro-type-cstyle-cast)
static const std::unordered_map<std::string_view, CppFnPtrTy> DispatchMap = {
#define DISPATCH_API(name, type)                                               \
  {#name, (CppFnPtrTy) static_cast<type>(&CppImpl::name)},
    CPPINTEROP_API_TABLE
#undef DISPATCH_API
};
// NOLINTEND(cppcoreguidelines-pro-type-cstyle-cast)

CppFnPtrTy CppGetProcAddress(const char* funcName) {
  auto it = DispatchMap.find(funcName);
  if (it == DispatchMap.end()) {
    std::cerr
        << "[CppInterOp Dispatch] Failed to find API: " << funcName
        << " May need to be ported to the symbol-address table in Dispatch.h\n";
    return nullptr;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return reinterpret_cast<CppFnPtrTy>(it->second);
}
