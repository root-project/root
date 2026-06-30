#include <CppInterOp/Dispatch.h>
#if __has_include("CppInterOp/CppInterOpAPI.inc")
using namespace CppImpl;
#define CPPINTEROP_API_FUNC(DN, CN, Ret, DeclArgs, CallArgs, RawTypes)  \
  Ret (*CppInternal::DispatchRaw::DN) RawTypes = nullptr;
#include "CppInterOp/CppInterOpAPI.inc"
#else
#define DISPATCH_API(name, type) CppAPIType::name Cpp::name = nullptr;
CPPINTEROP_API_TABLE
#undef DISPATCH_API
#endif
